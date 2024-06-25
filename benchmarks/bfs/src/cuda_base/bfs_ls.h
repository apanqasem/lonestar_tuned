#define BFS_VARIANT "lonestar"
#include "lonestargpu.h"
#include "cutil_subset.h"

//#define OPTIMIZED
//#define TILED

/* 
 * Placement Scheme 
 *
 * 1. HOST   : graph, dist, changed 
 * 2. IN     : graph
 * 3. OUT    : dist, changed
 * 4. SELECT : graph, dist
 * 5. DEV    : 
 */ 
 
__global__
void initialize(foru *dist, unsigned int nv) {
  unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < nv) {
    dist[ii] = MYINFINITY;
  }
}


__device__
bool processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;

	foru wt = 1;

	foru altdist = dist[src] + wt;
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
bool processnode(foru *dist, Graph &graph, unsigned work) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;
	if (nn >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(nn);
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
			changed = true;
		}
	}
	return changed;
}

#ifdef TILED 
__global__
void drelax_tiled(foru *dist, Graph graph, bool *changed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);

	do {
	  (*changed) = false;
	  for (unsigned ii = start; ii < end; ++ii) {
	    if (processnode(dist, graph, ii)) {
	      *changed = true;
	    }
	  }
	} while ((*changed));
}
#endif

__global__
void
#ifdef LAUNCH
__launch_bounds__(ML_MAX_THRDS_PER_BLK, ML_MIN_BLKS_PER_MP)
#endif
drelax(foru *dist, Graph graph, bool *changed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);

	for (unsigned ii = start; ii < end; ++ii) {
	  if (processnode(dist, graph, ii)) {
	    *changed = true;
	  }
	}
}

void bfs(Graph &graph, foru *dist)
{
  //	cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);

	foru foruzero = 0.0;
	KernelConfig kconf;
	double starttime, endtime;
	bool *changed, hchanged;
	int iteration = 0;

	kconf.setMaxThreadsPerBlock();
#ifdef ML
	kconf.setBlockSize(__BLOCKSIZE);
#endif
	kconf.setProblemSize(graph.nnodes);

	//	printf("initializing.\n");
	assert(kconf.coversProblem(0));
	starttime = rtclock();
	initialize <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph.nnodes);
#if defined HOST || OUT || SELECT
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
	endtime = rtclock();
	double init_time = 1000 * (endtime - starttime);
	CudaTest("initializing failed");
	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

#if defined HOST || OUT
	if (cudaMallocManaged(&changed, sizeof(bool)) != cudaSuccess)
	  CudaTest("allocating changed failed");
#else
	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess)
	  CudaTest("allocating changed failed");
#endif
	//	printf("solving.\n");
	starttime = rtclock();
#ifdef OPTIMIZED
#ifdef TILED
	drelax_tiled <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>>
	  (dist, graph, changed);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CudaTest("solving failed");
#else 
	do {
	  ++iteration;
	  (*changed) = false;
	  drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);
	  CUDA_SAFE_CALL(cudaDeviceSynchronize());
	  CudaTest("solving failed");
	} while ((*changed));
#endif
#else
#if defined HOST || OUT
	do {
	  ++iteration;
	  (*changed) = false;
          printf("Launch = %d %d\n", kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads());
	  drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);
	  CUDA_SAFE_CALL(cudaDeviceSynchronize());
	  CudaTest("solving failed");
	} while ((*changed));
#else 
	do {
	  ++iteration;
	  hchanged = false;
	  CUDA_SAFE_CALL(cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice));
          printf("Launch = %d %d\n", kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads());
	  drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);
	  CudaTest("solving failed");
	  
	  CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
	} while (hchanged);
#endif
#endif
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();

	//	printf("iterations = %d\n", iteration);
	//	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, 1000 * (endtime - starttime));
	//	printf("%3.4f,%3.4f", 1000 * (endtime - starttime),init_time);
}
