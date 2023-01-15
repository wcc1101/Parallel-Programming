#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <queue>
#include <list>
#include <algorithm>
#include <unistd.h>
#include <mpi.h>
#include <pthread.h>

#define REQUEST_MAP 0
#define DISPATCH_MAP 1 
#define FINISH_MAP 2
#define REQUEST_REDUCE 3
#define DISPATCH_REDUCE 4
#define FINISH_REDUCE 5

#define JOB_TRACKER 0

typedef std::pair<int, int> taskInfo;

using namespace std;

string JOB_NAME, INPUT_FILENAME, LOCALITY_CONFIG_FILENAME, OUTPUT_DIR;
int NUM_REDUCER, DELAY, CHUNK_SIZE;
int nNodes, rankId, nThreads;
int nTasks;

queue<pair<int, int>> tasks;
queue<pair<int, pair<int, int>>> complete;

pthread_mutex_t mutex;
pthread_cond_t cond;
pthread_mutex_t mutex_complete;
pthread_cond_t cond_complete;

static bool cmp(pair<string, int> a, pair<string, int> b) {
    // Modify here to change sorting function
    return a.first < b.first;
}

struct mapCmp {
    bool operator() (const string& a, const string& b) const {
        // Modify here to change sorting function
        return a < b;
    }
};

int diff(struct timespec start_time, struct timespec end_time) {
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

void JobTracker(void) {
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    ofstream logFile(OUTPUT_DIR + JOB_NAME + "-log.out");
    logFile << time(nullptr) << ",Start_Job," << nNodes << "," << nThreads << "," << JOB_NAME << "," << NUM_REDUCER << "," << DELAY << "," << INPUT_FILENAME << "," << CHUNK_SIZE << "," << LOCALITY_CONFIG_FILENAME << "," << OUTPUT_DIR << endl;

    list<taskInfo> mapTasks;
    int request, nChunks = 0, numPairs = 0, task[2], completeInfo[3], taskDone[2] = {-1, -1};

    // Read config
    ifstream configFile(LOCALITY_CONFIG_FILENAME);
    string line;
    while (getline(configFile, line)) {
        size_t pos = line.find(" ");
        int chunkID = stoi(line.substr(0, pos));
        int nodeID = stoi(line.substr(pos + 1)) % (nNodes - 1) + 1;
        taskInfo tmp = make_pair(chunkID, nodeID);
        mapTasks.push_back(tmp);
        nChunks++;
    }
    configFile.close();

    /* Map Scheduler */
    // Dispatch tasks to mappers
    bool isDispatch;
    while (!mapTasks.empty()) {
        isDispatch = false;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (auto mapTask = mapTasks.begin(); mapTask != mapTasks.end(); mapTask++) {
            if (mapTask->second == request) {
                task[0] = mapTask->first;
                task[1] = mapTask->second;
                logFile << time(nullptr) << ",Dispatch_MapTask," << task[0] << "," << request << endl;
                MPI_Send(&task, 2, MPI_INT, request, DISPATCH_MAP, MPI_COMM_WORLD);
                mapTasks.erase(mapTask);
                isDispatch = true;
                break;
            }
        }
        if (isDispatch == false) { // not found
            task[0] = mapTasks.begin()->first;
            task[1] = mapTasks.begin()->second;
            logFile << time(nullptr) << ",Dispatch_MapTask," << task[0] << "," << request << endl;
            MPI_Send(&task, 2, MPI_INT, request, DISPATCH_MAP, MPI_COMM_WORLD);
            mapTasks.erase(mapTasks.begin());
        }
    }

    // Tell mappers tasks are all dispatched
    for (int i = 0; i < nNodes - 1; i++) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&taskDone, 2, MPI_INT, request, DISPATCH_MAP, MPI_COMM_WORLD);
    }

    // Receive completeInfo
    for (int i = 0; i < nChunks; i++) {
        MPI_Recv(&completeInfo, 3, MPI_INT, MPI_ANY_SOURCE, FINISH_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        numPairs += completeInfo[2];
        logFile << time(nullptr) << ",Complete_MapTask," << completeInfo[0] << "," << completeInfo[1] << endl;
    }

    /* Shuffle */
    logFile << time(nullptr) << ",Start_Shuffle," << to_string(numPairs) << endl;

    struct timespec start_shuffle_time, end_shuffle_time;
    clock_gettime(CLOCK_MONOTONIC, &start_shuffle_time);

    // Open intermediate files
    ofstream *intermediateFiles = new ofstream[NUM_REDUCER];
    for (int i = 0; i < NUM_REDUCER; i++) {
        intermediateFiles[i] = ofstream("./intermediate_reducer_" + to_string(i + 1) + ".out");
    }

    // Read mapping results and write to reducers' interFiles
    for (int i = 0; i < nChunks; i++) {
        vector<pair<string, int>> interData;
        ifstream interFile("./intermediate" + to_string(i + 1) + ".out");
        string line;
        while (getline(interFile, line)) {
            size_t pos = line.find(" ");
            string word = line.substr(0, pos);
            int count = stoi(line.substr(pos + 1));
            interData.push_back(make_pair(word, count));
        }
        for (auto pair: interData) {
            // Modify here to change partition function
            int idx = (int)pair.first[0] % NUM_REDUCER;
            intermediateFiles[idx] << pair.first << " " << pair.second << endl;
        }
        interFile.close();
    }
    for (int i = 0; i < NUM_REDUCER; i++)
        intermediateFiles[i].close();

    clock_gettime(CLOCK_MONOTONIC, &end_shuffle_time);
    logFile << time(nullptr) << ",Finish_Shuffle," << to_string(diff(start_shuffle_time, end_shuffle_time)) << endl;

    /* Reduce Scheduler */
    // Dispatch tasks to reducers
    for (int i = 1; i <= NUM_REDUCER; i++) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        logFile << time(nullptr) << ",Dispatch_ReduceTask," << i << "," << request << endl;
        MPI_Send(&i, 1, MPI_INT, request, DISPATCH_REDUCE, MPI_COMM_WORLD);
    }

    // Tell reducers tasks are all dispatched
    for (int i = 0; i < nNodes - 1; i++){
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&taskDone, 1, MPI_INT, request, DISPATCH_REDUCE, MPI_COMM_WORLD);
    }

    // Receive completeInfo
    for (int i = 0; i < NUM_REDUCER; i++) {
        MPI_Recv(&completeInfo, 2, MPI_INT, MPI_ANY_SOURCE, FINISH_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        logFile << time(nullptr) << ",Complete_ReduceTask," << completeInfo[0] << "," << completeInfo[1] << endl;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    logFile << time(nullptr) << ",Finish_Job," << to_string(diff(start_time, end_time)) << endl;
    logFile.close();
}

void* mapperCal(void* args) {
    struct timespec start_time, end_time, temp;
    double exe_time;
    // bool init_state = true, isEmpty = false;

    while (1) {
        pair<int, int> task;

        // Waiting until the other thread done
        pthread_mutex_lock(&mutex);
        while (tasks.empty()) {
            pthread_cond_wait(&cond, &mutex);
        }
        task = tasks.front();
        tasks.pop();
        pthread_mutex_unlock(&mutex);

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Reading from a remote location, sleep for DELAY seconds
        if (task.second != rankId) {
            sleep(DELAY);
        }

        // Input Split function
        map<int, string> records;
        ifstream inputFile(INPUT_FILENAME);
        int count = 0;
        string line;
        // Find the position
        while (count != (task.first - 1) * CHUNK_SIZE && getline(inputFile, line)) {
            count++;
        }
        while (count != task.first * CHUNK_SIZE && getline(inputFile, line)) {
            records[count++] = line;
        }
        inputFile.close();

        // Map function
        map<string, int> map_output;
        size_t pos = 0;
        for (auto record: records) {
            while ((pos = record.second.find(" ")) != string::npos) {
                string word = record.second.substr(0, pos);
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
        ofstream out("./intermediate" + to_string(task.first) + ".out");
        for (auto it: map_output) {
            out << it.first << " " << it.second << endl;
        }
        out.close();

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        // Write into Complete queue
        pthread_mutex_lock(&mutex_complete);
        complete.push(make_pair(task.first, make_pair(diff(start_time, end_time), map_output.size())));
        nTasks--;
        pthread_mutex_unlock(&mutex_complete);
        pthread_cond_signal(&cond_complete);
    }
}

void Mapper(void) {
    pthread_t threads[nThreads - 1];
    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(&cond, nullptr);
    pthread_mutex_init(&mutex_complete, nullptr);
    pthread_cond_init(&cond_complete, nullptr);

    int task[2], completeInfo[3];

    // Create (s-1) threads to do mapping
    for (int i = 0; i < nThreads-1; i++) {
        pthread_create(&threads[i], nullptr, &mapperCal, nullptr);
    }

    nTasks = 0;
    // Receive tasks
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        while (nTasks == nThreads - 1) {
            pthread_cond_wait(&cond_complete, &mutex_complete);
        }
        pthread_mutex_unlock(&mutex_complete);
        MPI_Send(&rankId, 1, MPI_INT, 0, REQUEST_MAP, MPI_COMM_WORLD);
        MPI_Recv(&task, 2, MPI_INT, 0, DISPATCH_MAP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (task[0] == -1) { // End
            break;
        }
        pthread_mutex_lock(&mutex_complete);
        nTasks++;
        pthread_mutex_unlock(&mutex_complete);
        pthread_mutex_lock(&mutex);
        tasks.push(make_pair(task[0], task[1]));
        pthread_mutex_unlock(&mutex);
        pthread_cond_signal(&cond);
    }

    // Send completeInfo
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        if (complete.empty() && nTasks == 0) {    // if all mapTasks are finished and all information is sent
            pthread_mutex_unlock(&mutex_complete);
            break;
        }
        else if (!complete.empty()) {   // send information
            pair<int, pair<int, int>> info = complete.front();
            complete.pop();
            pthread_mutex_unlock(&mutex_complete);
            completeInfo[0] = info.first;
            completeInfo[1] = info.second.first;
            completeInfo[2] = info.second.second;
            MPI_Send(&completeInfo, 3, MPI_INT, 0, FINISH_MAP, MPI_COMM_WORLD);
        }
        else {
            pthread_mutex_unlock(&mutex_complete);
        }
    }
}

void Reducer(void) {
    struct timespec start_time, end_time, temp;
    double exe_time;
    int task[2], completeInfo[3];
    queue<pair<int, int>> reduce_job_time;

    // Receive tasks
    while (true) {
        MPI_Send(&rankId, 1, MPI_INT, JOB_TRACKER, REQUEST_REDUCE, MPI_COMM_WORLD);
        MPI_Recv(&task, 1, MPI_INT, JOB_TRACKER, DISPATCH_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (task[0] == -1) { // End
            break;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Read from inter files
        vector<pair<string, int>> data;
        ifstream interFile("./intermediate_reducer_" + to_string(task[0]) + ".out");
        string line;
        while (getline(interFile, line)) {
            size_t pos = line.find(" ");
            string word = line.substr(0, pos);
            int count = stoi(line.substr(pos+1));
            data.push_back(make_pair(word, count));
        }
        interFile.close();

        // Sort function
        sort(data.begin(), data.end(), cmp);
 
        // Group function
        map<string, vector<int>, mapCmp>groupData;
        for (auto it : data) {
            string key;
            // Modify here to change group function
            key = it.first;
            groupData[key].push_back(it.second);
        }

        // Reduce function
        vector<pair<string, int>>reduce_result;
        for (auto group : groupData) {
            int sum = 0;
            for (auto it : group.second) {
                sum += it;
            }
            reduce_result.push_back(make_pair(group.first, sum));
        }

        // Output function
        ofstream outputFile(OUTPUT_DIR + JOB_NAME + "-" + to_string(task[0]) + ".out");
        for (auto r : reduce_result)
            outputFile << r.first << " " << r.second << endl;
        outputFile.close();

        clock_gettime(CLOCK_MONOTONIC, &end_time);
        reduce_job_time.push(make_pair(task[0], diff(start_time, end_time)));
    }

    // Send completeInfo
    while (!reduce_job_time.empty()) {
        pair<int, int> info = reduce_job_time.front();
        reduce_job_time.pop();
        completeInfo[0] = info.first;
        completeInfo[1] = info.second;
        MPI_Send(&completeInfo, 2, MPI_INT, JOB_TRACKER, FINISH_REDUCE, MPI_COMM_WORLD);
    }
}

void TaskTracker(void) {
    Mapper();
    Reducer();
}

int main(int argc, char **argv) {
    JOB_NAME = string(argv[1]);
    NUM_REDUCER = stoi(argv[2]);
    DELAY = stoi(argv[3]);
    INPUT_FILENAME = string(argv[4]);
    CHUNK_SIZE = stoi(argv[5]);
    LOCALITY_CONFIG_FILENAME = string(argv[6]);
    OUTPUT_DIR = string(argv[7]);

    // MPI initialize
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    // thread initialize
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    nThreads = CPU_COUNT(&cpu_set);

    // start MapReduce
    if (rankId == JOB_TRACKER) {
        JobTracker();
    }
    else {
        TaskTracker();
    }

    MPI_Finalize();
    return 0;
}
