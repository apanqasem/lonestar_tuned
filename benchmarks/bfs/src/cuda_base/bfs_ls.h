#define BFS_VARIANT "lonestar"
#include "cutil_subset.h"



__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}


__device__
bool processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst, unsigned *traversed_edges) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;
	//	if (graph.getDestination(src, ii) >= graph.nnodes) return 0;

	//	(*traversed_edges)++;
	//	foru wt = 1;

	foru altdist = dist[src] + 1;
	//	if (altdist < dist[graph.getDestination(src, ii)]) {
	if (altdist < dist[dst]) {
	  foru olddist = atomicMin(&dist[dst], altdist);
	  if (altdist < olddist) { 
		 return true;
	  } 
	  // someone else updated distance to a lower value.
	}
	return false;
}
__device__
bool processnode(foru *dist, Graph &graph, unsigned work, unsigned *traversed_edges) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  //	unsigned nn = work;
	if (work >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(work);
	#pragma unroll (16)
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
	  //  	  unsigned dst = graph.nnodes;
	  //	  foru olddist = processedge(dist, graph, nn, ii, graph.nnodes, traversed_edges);
	  //	  if (olddist) {

	  if (processedge(dist, graph, work, ii, graph.nnodes, traversed_edges)) {
	    changed = true;
	  }
	}
	return changed;
}

__global__ void 
#ifdef LAUNCH
__launch_bounds__(ML_MAX_THRDS_PER_BLK, ML_MIN_BLKS_PER_MP)
#endif
     drelax(foru *dist, Graph graph, bool *changed, unsigned *traversed_edges) {
  	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);
	
	#pragma unroll (2)
	for (unsigned ii = start; ii < end; ++ii) {
	  if (processnode(dist, graph, ii, traversed_edges)) {
	    *changed = true;
	  }
	}
}


void bfs(Graph &graph, foru *dist)
{
	cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);

	foru foruzero = 0.0;
	KernelConfig kconf;
	double starttime, endtime;
	bool *changed, hchanged;
	int iteration = 0;
	unsigned *traversed_edges, traversed_edges_host;

	kconf.setProblemSize(graph.nnodes);

	kconf.setMaxThreadsPerBlock();
#ifdef ML
	kconf.setBlockSize(__BLOCKSIZE0);
	kconf.setNumberOfBlocks((graph.nnodes + __BLOCKSIZE0 - 1) / __BLOCKSIZE0 );
#endif
	printf("initializing.\n");
	assert(kconf.coversProblem(0));
	initialize <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph.nnodes);
	CudaTest("initializing failed");

	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");
	if (cudaMalloc((void **)&traversed_edges, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating changed failed");

	printf("solving.\n");
	starttime = rtclock();
	
        traversed_edges_host = 0;
	do {
		++iteration;
		hchanged = false;

		CUDA_SAFE_CALL(cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice));
#ifdef TEPS		
		CUDA_SAFE_CALL(cudaMemcpy(traversed_edges, &traversed_edges_host, sizeof(traversed_edges_host), cudaMemcpyHostToDevice));
#endif
		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed, traversed_edges);
		CudaTest("solving failed");

		CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
#ifdef TEPS
		CUDA_SAFE_CALL(cudaMemcpy(&traversed_edges_host, traversed_edges, sizeof(traversed_edges_host), cudaMemcpyDeviceToHost)); 
#endif
	} while (hchanged);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, 1000 * (endtime - starttime));
	//	printf("\ttraversed edges = %u\n", traversed_edges_host);
}
