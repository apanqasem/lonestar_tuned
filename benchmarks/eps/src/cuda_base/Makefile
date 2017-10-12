EXE=eps

# ifeq ($(reg), def)
# 	COMMFLAGS=-O3 --compiler-options -Wall -Xptxas -$(opt)
# else
# 	COMMFLAGS=-O3 --compiler-options -Wall -Xptxas -$(opt) --maxrregcount=$(reg)
# endif 

# #NVCC= "$(shell which nvcc)"

# FLAGS= -arch=sm_35  ${COMMFLAGS} -Xptxas -v #-Xptxas -dlcm=cg#disable l1 cache
FLAGS+= -ccbin=g++ -Xcompiler -fopenmp

include ../../../Makefile.conf

ifeq ($(enable_monitor), 1)
	FLAGS+= -DENABLE_MONITORING
endif


ifeq ($(enable_check), 1)
	FLAGS+= -DENABLE_CHECKING
endif

OBJS=  	main.o 
DEPS= 	Makefile \
		expander.cuh \
		inspector.cuh \
		comm.h \
		bfs_gpu_opt.cuh \
		wtime.h \
		write_result.cuh \
		scan.cuh \
		allocator.cuh 

%.o:%.cu $(DEPS)
	${NVCC} -c  ${FLAGS} $< -o $@

${EXE}:${OBJS}
	${NVCC} ${OBJS} $(FLAGS) -o ${EXE}

clean:
	rm -rf results.dat *.gen *~ *.o ${EXE}
