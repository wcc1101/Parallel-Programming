#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <queue>
#include <list>
#include <algorithm>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>

#define REQUEST_MAP 0
#define DISPATCH_MAP 1 
#define FINISH_MAP 2
#define REQUEST_REDUCE 3
#define DISPATCH_REDUCE 4
#define FINISH_REDUCE 5

// #define (size - 1) 4

using namespace std;

string job_name, input_filename, locality_config_filename, output_dir;
int num_reducer, delay, chunk_size;
int nThreads;
int num_chunk;
int num_jobs;
int size;

queue<pair<int, int>> tasks;
queue<pair<int, pair<int, int>>> complete;

pthread_mutex_t mutex;
pthread_cond_t cond;
pthread_mutex_t mutex_complete;
pthread_cond_t cond_complete;

// Modify here to change sorting function
bool ascending = true;
static bool comp(pair<string, int> a, pair<string, int> b) {
    if (ascending == true)
        return (a.first < b.first);
    else 
        return (a.first > b.first);
}

struct classcomp {
    bool operator() (const string& a, const string& b) const {
        if (ascending == true)
            return a < b;
        else 
            return a > b;
    }
};

int calc_time(struct timespec start_time, struct timespec end_time) {
    struct timespec temp;
    if ((end_time.tv_nsec - start_time.tv_nsec) < 0) {
        temp.tv_sec = end_time.tv_sec-start_time.tv_sec-1;
        temp.tv_nsec = 1000000000 + end_time.tv_nsec - start_time.tv_nsec;
    } else {
        temp.tv_sec = end_time.tv_sec - start_time.tv_sec;
        temp.tv_nsec = end_time.tv_nsec - start_time.tv_nsec;
    }
    double exe_time = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    return (int)(exe_time+0.5);
}

void jobTracker(int rank, int size) {
    struct timespec start_time, end_time;
    list<pair<int, int>> mapTasks;   // job queue
    int request_node;
    int job_send[2];    // (chunkID, nodeID)
    int cnt;
    int complete_info[3];   // (job, time, pairs)
    int total_pairs = 0;
    string line;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    ofstream out(output_dir + job_name + "-log.out");
    out << time(nullptr) << ",Start_Job," << size << "," << nThreads << "," << job_name << "," << num_reducer << "," << delay << "," << input_filename << "," << chunk_size << "," << locality_config_filename << "," << output_dir << endl;

    ifstream input_file(locality_config_filename);
    while (getline(input_file, line)) {
        size_t pos = line.find(" ");
        int chunkID = stoi(line.substr(0, pos));
        int nodeID = stoi(line.substr(pos+1)) % (size - 1);
        mapTasks.push_back(make_pair(chunkID, nodeID));    // Locality information: (chunkID, nodeID)
    }
    input_file.close();

    num_chunk = mapTasks.size();

    bool isDispatch;
    while (!mapTasks.empty()) {
        isDispatch = false;
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (auto job = mapTasks.begin(); job != mapTasks.end(); job++) {
            if (job->second == request_node) {
                job_send[0] = job->first;
                job_send[1] = job->second;
                out << time(nullptr) << ",Dispatch_MapTask," << job_send[0] << "," << request_node << endl;
                MPI_Send(&job_send, 2, MPI_INT, request_node, DISPATCH_MAP, MPI_COMM_WORLD);
                mapTasks.erase(job);
                isDispatch = true;
                break;
            }
        }
        if (isDispatch == false) {
            job_send[0] = mapTasks.begin()->first;
            job_send[1] = mapTasks.begin()->second;
            out << time(nullptr) << ",Dispatch_MapTask," << job_send[0] << "," << request_node << endl;
            MPI_Send(&job_send, 2, MPI_INT, request_node, DISPATCH_MAP, MPI_COMM_WORLD);
            mapTasks.erase(mapTasks.begin());
        }
    }

    int jobDone[2] = {-1, -1};
    for (int i = 0; i < size - 1; i++) {
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&jobDone, 2, MPI_INT, request_node, DISPATCH_MAP, MPI_COMM_WORLD);
    }

    for (int i = 0; i < num_chunk; i++) {
        MPI_Recv(&complete_info, 3, MPI_INT, MPI_ANY_SOURCE, FINISH_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total_pairs += complete_info[2];
        out << time(nullptr) << ",Complete_MapTask," << complete_info[0] << "," << complete_info[1] << endl;
    }

    out << time(nullptr) << ",Start_Shuffle," << to_string(total_pairs) << endl;

    struct timespec start_shuffle_time, end_shuffle_time;
    clock_gettime(CLOCK_MONOTONIC, &start_shuffle_time);

    ofstream *files = new ofstream[num_reducer];
    for (int i = 0; i < num_reducer; i++) {
        files[i] = ofstream(output_dir + job_name + "-intermediate_reducer_" + to_string(i+1) + ".out");
    }

    vector<pair<string, int>> data;
    for (int i = 0; i < num_chunk; i++) {
        data.clear();
        ifstream input_file(output_dir + job_name + "-intermediate" + to_string(i+1) + ".out");
        string line;
        while (getline(input_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos+1));
            data.push_back(make_pair(key, value));
        }
        // Partition function
        for (auto it: data) {
            int idx = (it.first[0] - 'A') % num_reducer;
            files[idx] << it.first << " " << it.second << endl;
        }
    }

    for (int i = 0; i < num_reducer; i++)
        files[i].close();
    
    clock_gettime(CLOCK_MONOTONIC, &end_shuffle_time);
    out << time(nullptr) << ",Finish_Shuffle," << to_string(calc_time(start_shuffle_time, end_shuffle_time)) << endl;

    for (int i = 1; i <= num_reducer; i++) {
        out << time(nullptr) << ",Dispatch_ReduceTask," << i << "," << request_node << endl;
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&i, 1, MPI_INT, request_node, DISPATCH_REDUCE, MPI_COMM_WORLD);
    }

    for (int i = 0; i < size - 1; i++){
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&jobDone, 1, MPI_INT, request_node, DISPATCH_REDUCE, MPI_COMM_WORLD);
    }

    for (int i = 0; i < num_reducer; i++) {
        MPI_Recv(&complete_info, 2, MPI_INT, MPI_ANY_SOURCE, FINISH_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        out << time(nullptr) << ",Complete_ReduceTask," << complete_info[0] << "," << complete_info[1] << endl;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    out << time(nullptr) << ",Finish_Job," << to_string(calc_time(start_time, end_time)) << endl;
}

void* mapCal(void* arg) {
    int rank = *(int *)arg;
    struct timespec start_time, end_time, temp;
    double exe_time;
    bool init_state = true;

    while (true) {
        pair<int, int> task;

        pthread_mutex_lock(&mutex);
        if (tasks.empty()) {
            if (init_state == false) // done
                break;
            else { // waiting
                pthread_mutex_unlock(&mutex);
                continue;
            }
        }
        init_state = false;
        task = tasks.front();
        tasks.pop();
        pthread_mutex_unlock(&mutex);

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // read from a remote location
        if (task.second != rank) {
            sleep(delay);
        }

        // Input Split function
        map<int, string> records;
        ifstream input_file(input_filename);
        int count = 0;
        string line;
        while (count != (task.first - 1) * chunk_size && getline(input_file, line)) {
            count++;
        }
        while (count != task.first * chunk_size && getline(input_file, line)) {
            records[count++] = line;
        }

        // Map function
        map<string, int> map_output;
        size_t pos = 0;
        string word;
        for (auto record : records) {
            while ((pos = record.second.find(" ")) != string::npos) {
                word = record.second.substr(0, pos);
                if (map_output.count(word) == 0)
                    map_output[word] = 1;
                else
                    map_output[word]++;
                record.second.erase(0, pos + 1);
            }
            // Last one
            if (map_output.count(record.second) == 0)
                map_output[record.second] = 1;
            else
                map_output[record.second]++;
        }

        // Write intermediate result
        ofstream out(output_dir + job_name + "-intermediate" + to_string(task.first) + ".out");
        for (auto it: map_output) {
            out << it.first << " " << it.second << endl;
        }
        out.close();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        pthread_mutex_lock(&mutex_complete);
        complete.push(make_pair(task.first, make_pair(calc_time(start_time, end_time), map_output.size())));
        num_jobs--;
        pthread_mutex_unlock(&mutex_complete);
        pthread_cond_signal(&cond_complete);
    }
}

void mapper(int rank) {
    pthread_t threads[nThreads - 1];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_mutex_init(&mutex_complete, NULL);
    pthread_cond_init(&cond_complete, NULL);

    int arg[nThreads - 1];
    for (int i = 0; i < nThreads - 1; i++) {
        arg[i] = rank;
        pthread_create(&threads[i], NULL, &mapCal, (void *)&arg[i]);
    }

    num_jobs = 0;

    int job[2] = {0};
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        while (num_jobs == nThreads-1) {
            pthread_cond_wait(&cond_complete, &mutex_complete);
        }
        pthread_mutex_unlock(&mutex_complete);

        MPI_Send(&rank, 1, MPI_INT, (size - 1), REQUEST_MAP, MPI_COMM_WORLD);
        MPI_Recv(&job, 2, MPI_INT, (size - 1), DISPATCH_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (job[0] == -1) { // End
            break;
        }
        pthread_mutex_lock(&mutex_complete);
        num_jobs++;
        tasks.push(make_pair(job[0], job[1]));
        pthread_mutex_unlock(&mutex_complete);
        cout << "worker" << rank << " get job" << job[0] << " stored at device" << job[1] << endl;
    }

    int complete_info[3];
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        if (complete.empty() && num_jobs == 0) {    // if all jobs are finished and all information is sent
            pthread_mutex_unlock(&mutex_complete);
            break;
        }
        else if (!complete.empty()) {   // send information
            pair<int, pair<int, int>> info = complete.front();
            complete.pop();
            pthread_mutex_unlock(&mutex_complete);
            complete_info[0] = info.first;
            complete_info[1] = info.second.first;
            complete_info[2] = info.second.second;
            MPI_Send(&complete_info, 3, MPI_INT, (size - 1), FINISH_MAP, MPI_COMM_WORLD);
        }
        else {
            pthread_mutex_unlock(&mutex_complete);
        }
    }
}

void reducer(int rank) {
    struct timespec start_time, end_time, temp;
    double exe_time;
    int job[2] = {0};
    int complete_info[3];
    string line;
    queue<pair<int, int>> reduce_job_time;
    while (true) {
        MPI_Send(&rank, 1, MPI_INT, (size - 1), REQUEST_REDUCE, MPI_COMM_WORLD);
        MPI_Recv(&job, 1, MPI_INT, (size - 1), DISPATCH_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (job[0] == -1) { // termination condition, all tasks are dispatched
            break;
        }
        cout << "worker" << rank << " get reduce job" << job[0] << endl;
        
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        vector<pair<string, int>> data;
        ifstream input_file(output_dir + job_name + "-intermediate_reducer_" + to_string(job[0]) + ".out");

        while (getline(input_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos+1));
            data.push_back(make_pair(key, value));
        }
        input_file.close();

        // Sort function
        sort(data.begin(), data.end(), comp);

        // Group function
        map<string, vector<int>, classcomp>grouped_data;
        for (auto it : data) {
            grouped_data[it.first].push_back(it.second);
        }

        // Reduce function
        map<string, int>reduce_result;
        for (auto group : grouped_data) {
            int sum = 0;
            for (auto it : group.second) {
                sum += it;
            }
            reduce_result[group.first] = sum;
        }

        // Output function
        ofstream output_file(output_dir + job_name + "-" + to_string(job[0]) + ".out");
        for (auto r : reduce_result)
            output_file << r.first << " " << r.second << endl;
        output_file.close();

        clock_gettime(CLOCK_MONOTONIC, &end_time);
        reduce_job_time.push(make_pair(job[0], calc_time(start_time, end_time)));
    }

    while (!reduce_job_time.empty()) {
        pair<int, int> info = reduce_job_time.front();
        reduce_job_time.pop();
        complete_info[0] = info.first;
        complete_info[1] = info.second;
        MPI_Send(&complete_info, 2, MPI_INT, (size - 1), FINISH_REDUCE, MPI_COMM_WORLD);
    }
}

void taskTracker(int rank) {
    mapper(rank);
    reducer(rank);
}

int main(int argc, char **argv) {
    // Get command arguments
    job_name = string(argv[1]);
    num_reducer = stoi(argv[2]);
    delay = stoi(argv[3]);
    input_filename = string(argv[4]);
    chunk_size = stoi(argv[5]);
    locality_config_filename = string(argv[6]);
    output_dir = string(argv[7]);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get number of threads
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    nThreads = CPU_COUNT(&cpu_set);

    if (rank == (size - 1)) {
        jobTracker(rank, size);
    } else {
        taskTracker(rank);
    }

    MPI_Finalize();
    return 0;
}