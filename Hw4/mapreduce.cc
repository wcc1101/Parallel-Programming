#include <iostream>
#include <fstream>
#include <mpi.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <list>
#include <queue>
#include <map>
#include <algorithm>

#include <chrono>

using namespace std;

#define REQUEST_MAP 0
#define DISPATCH_MAP 1 
#define FINISH_MAP 2
#define REQUEST_REDUCE 3
#define DISPATCH_REDUCE 4
#define FINISH_REDUCE 5

typedef pair<string, int> Item;

// varaiables
string JOB_NAME, INPUT_FILENAME, LOCALITY_CONFIG_FILENAME, OUTPUT_DIR;
int NUM_REDUCER, DELAY, CHUNK_SIZE;
// int rank, size;
int nThreads;
int numChunk;
pthread_mutex_t mutex;
pthread_cond_t cond;
pthread_mutex_t mutex_complete;
pthread_cond_t cond_complete;

queue<int> tasks;

int num_jobs;

bool ascending = true;
static bool comp(Item a, Item b) {
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

void jobTracker(int rank, int size) {
    // output log
    ofstream log_file(OUTPUT_DIR + JOB_NAME + "-log.out");

    // read config
    ifstream localityConfig(LOCALITY_CONFIG_FILENAME);
    list<pair<int, int> > mapTasks;
    string line;
    int complete_info[3];
    while (getline(localityConfig, line)){
        size_t pos = line.find(" ");
        int chunkID = stoi(line.substr(0, pos));
        int nodeID = stoi(line.substr(pos + 1)) % (size - 1) + 1;
        mapTasks.push_back(make_pair(chunkID, nodeID));
    }
    numChunk = mapTasks.size();
    localityConfig.close();

    // dispatch mapTasks
    int request;
    bool isDispatch;
    while (!mapTasks.empty()) {
        isDispatch = false;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (auto task = mapTasks.begin(); task != mapTasks.end(); task++) {
            if (task->second == request) {
                log_file << time(nullptr) << ",Dispatch_MapTask," << task->first << "," << request << endl;
                MPI_Send(&(task->first), 1, MPI_INT, request, DISPATCH_MAP, MPI_COMM_WORLD);
                mapTasks.erase(task);
                isDispatch = true;
                break;
            }
        }
        if (isDispatch == false) { // no match
            log_file << time(nullptr) << ",Dispatch_MapTask," << mapTasks.begin()->first << "," << request << endl;
            MPI_Send(&(mapTasks.begin()->first), 1, MPI_INT, request, DISPATCH_MAP, MPI_COMM_WORLD);
            mapTasks.erase(mapTasks.begin());
        }
    }

    // mapper mapTasks done
    int taskDone = -1;
    for (int i = 1; i < size; i++) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&taskDone, 1, MPI_INT, request, DISPATCH_MAP, MPI_COMM_WORLD);
    }

    int total_pairs = 0;
    for (int i = 0; i < numChunk; i++) {
        MPI_Recv(&complete_info, 3, MPI_INT, MPI_ANY_SOURCE, FINISH_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total_pairs += complete_info[2];
        // out << time(nullptr) << ",Complete_MapTask," << complete_info[0] << "," << complete_info[1] << endl;
    }

    // Shuffle
    chrono::steady_clock::time_point start, end;
    start = chrono::steady_clock::now();
    log_file << time(nullptr) << ",Start_Shuffle," << to_string(total_pairs) << endl;

    ofstream* files = new ofstream[NUM_REDUCER];
    for(int i = 0; i < NUM_REDUCER; i++)
        files[i] = ofstream(OUTPUT_DIR + JOB_NAME + "-intermediate_reducer_" + to_string(i+1) + ".out");

    vector<Item> data;
    for (int i = 0; i < numChunk; i++){
        data.clear();
        ifstream inputFile(OUTPUT_DIR + JOB_NAME + "-intermediate" + to_string(i+1) + ".out");
        string line;
        while(getline(inputFile, line)){
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos+1));
            data.push_back(make_pair(key, value));
        }
        for (auto pair : data) {
            int idx = (pair.first[0] - 'A') % NUM_REDUCER;
            files[idx] << pair.first << " " << pair.second << endl;
        }
        inputFile.close();
    }

    for (int i = 0; i < NUM_REDUCER; i++)
        files[i].close();

    end = chrono::steady_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::milliseconds>(end - start);
    log_file << time(nullptr) << ",Finish_Shuffle," << ((double)time_span.count())/1000 << endl;

    for (int i = 1; i <= NUM_REDUCER; i++) {
        log_file << time(nullptr) << ",Dispatch_ReduceTask," << i << "," << request << endl;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&i, 1, MPI_INT, request, DISPATCH_REDUCE, MPI_COMM_WORLD);
    }

    for (int i = 1; i < size; i++){
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&taskDone, 1, MPI_INT, request, DISPATCH_REDUCE, MPI_COMM_WORLD);
    }

    for (int i = 0; i < numChunk; i++) {
        MPI_Recv(&complete_info, 2, MPI_INT, MPI_ANY_SOURCE, FINISH_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // out << time(nullptr) << ",Complete_ReduceTask," << complete_info[0] << "," << complete_info[1] << endl;
    }
}

void* pool(void* args) {
    struct timespec start_time, end_time, temp;
    double exe_time;
    bool init_state = true;

    while (true) {
        int task;

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

        // Input split
        map<int, string> records;
        ifstream input_file(INPUT_FILENAME);
        int count = 0;
        string line;
        for (int i = 0; i < (task - 1) * CHUNK_SIZE; i++) {
            getline(input_file, line);
            count++;
        }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            getline(input_file, line);
            records[count] = line;
            count++;
        }
        input_file.close();

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
        ofstream out(OUTPUT_DIR + JOB_NAME + "-intermediate" + to_string(task) + ".out");
        for (auto it : map_output) {
            out << it.first << " " << it.second << endl;
        }
        out.close();

        pthread_mutex_lock(&mutex_complete);
        num_jobs--;
        pthread_mutex_unlock(&mutex_complete);
        pthread_cond_signal(&cond_complete);
    }
    pthread_exit(NULL);
}

void mapper(int rank) {
    pthread_t threads[nThreads - 1];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_mutex_init(&mutex_complete, NULL);
    pthread_cond_init(&cond_complete, NULL);

    for (int i = 0; i < nThreads - 1; i++)
        pthread_create(&threads[i], NULL, &pool, NULL);

    num_jobs = 0;

    int dataChunk;
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        while (num_jobs == nThreads - 1) {
            pthread_cond_wait(&cond_complete, &mutex_complete);
        }
        pthread_mutex_unlock(&mutex_complete);

        MPI_Send(&rank, 1, MPI_INT, 0, REQUEST_MAP, MPI_COMM_WORLD);
        MPI_Recv(&dataChunk, 1, MPI_INT, 0, DISPATCH_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (dataChunk == -1) // End
            break;

        pthread_mutex_lock(&mutex_complete);
        num_jobs++;
        tasks.push(dataChunk);
        pthread_mutex_unlock(&mutex_complete);
    }

    int complete_info[3];
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        if (complete.empty() && num_jobs == 0) {    // if all jobs are finished and all information is sent
            pthread_mutex_unlock(&mutex_complete);
            break;
        }
        else if (!complete.empty()) {   // send information
            pair<int, pair<int, int> > info = complete.front();
            complete.pop();
            pthread_mutex_unlock(&mutex_complete);
            complete_info[0] = info.first;
            complete_info[1] = info.second.first;
            complete_info[2] = info.second.second;
            MPI_Send(&complete_info, 3, MPI_INT, 0, FINISH_MAP, MPI_COMM_WORLD);
        }
        else {
            pthread_mutex_unlock(&mutex_complete);
        }
    }
}

void reducer(int rank) {
    queue<pair<int, int>> reduce_job_time;
    while (true){
        int reducerID;
        MPI_Send(&rank, 1, MPI_INT , 0, REQUEST_REDUCE, MPI_COMM_WORLD);
        MPI_Recv(&reducerID, 1, MPI_INT, 0, DISPATCH_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (reducerID == -1) { // End
            break;
        }
        ifstream inter_file(OUTPUT_DIR + JOB_NAME + "-intermediate_reducer_" + to_string(reducerID) + ".out");
        string line;
        vector<Item> data;

        while (getline(inter_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos + 1));
            data.push_back(make_pair(key, value));
        }
        inter_file.close();

        // Sort function
        sort(data.begin(), data.end(), comp);

        // Group function
        map<string, vector<int> >group_pair;
        for (auto it : data) {
            group_pair[it.first].push_back(it.second);
        }

        // Reduce function
        map<string, int>result;
        for (auto group : group_pair) {
            int sum = 0;
            for (auto it : group.second) {
                sum += it;
            }
            result[group.first] = sum;
        }

        // Output function
        ofstream output_file(OUTPUT_DIR + JOB_NAME + "-" + to_string(reducerID) + ".out");
        for(auto r : result)
            output_file << r.first << " " << r.second << endl;
        output_file.close();

        reduce_job_time.push(make_pair(reducerID, 0));
    }
    int complete_info[3];
    while (!reduce_job_time.empty()) {
        pair<int, int> info = reduce_job_time.front();
        reduce_job_time.pop();
        complete_info[0] = info.first;
        complete_info[1] = info.second;
        MPI_Send(&complete_info, 2, MPI_INT, 0, FINISH_REDUCE, MPI_COMM_WORLD);
    }
}

void taskTracker(int rank) {
    mapper(rank);
    reducer(rank);
}

int main(int argc, char **argv) {
    // read command inputs
    JOB_NAME = string(argv[1]);
    NUM_REDUCER = atoi(argv[2]);
    DELAY = atoi(argv[3]);
    INPUT_FILENAME = string(argv[4]);
    CHUNK_SIZE = atoi(argv[5]);
    LOCALITY_CONFIG_FILENAME = string(argv[6]);
    OUTPUT_DIR = string(argv[7]);
    cout << JOB_NAME << " " << NUM_REDUCER << " " << DELAY << " " << INPUT_FILENAME << " " << CHUNK_SIZE << " " << LOCALITY_CONFIG_FILENAME << " " << OUTPUT_DIR << endl;

    // initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // get cpu set
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    nThreads = CPU_COUNT(&cpuset);

    if (rank == 0) {
        jobTracker(rank, size);
    }
    else {
        taskTracker(rank);
    }

    MPI_Finalize();

    return 0;
}