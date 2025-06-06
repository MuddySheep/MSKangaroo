// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"
#include "cuda_helpers.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode

int RCGpuKang::CalcKangCnt()
{
	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	return Kparams.BlockSize* Kparams.GroupCnt* Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	u64 total_mem = 0;
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;

        CUDA_CHECK_ERROR(cudaSetDevice(CudaIndex));

	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
	Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
	Kparams.IsGenMode = gGenMode;

//allocate gpu mem
	u64 size;

	if (!IsOldGpu)
	{
		//L2	
                int L2size = Kparams.KangCnt * (3 * 32);
                total_mem += L2size;
                CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.L2, L2size));
                size = L2size;
                if (size > persistingL2CacheMaxSize)
                        size = persistingL2CacheMaxSize;
                CUDA_CHECK_ERROR(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size)); // set max allowed size for L2

        if (!IsOldGpu)
        {
                //L2
                int L2size = Kparams.KangCnt * (3 * 32);
                total_mem += L2size;
                err = buf_L2.allocate(L2size);
                Kparams.L2 = buf_L2.get<u64>();
                if (err != cudaSuccess)
                {
                        printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                        return false;
                }
		size = L2size;
		if (size > persistingL2CacheMaxSize)
			size = persistingL2CacheMaxSize;
		err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2

		//persisting for L2
		cudaStreamAttrValue stream_attribute;                                                   
		stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
		stream_attribute.accessPolicyWindow.num_bytes = size;										
		stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
		stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
		stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
                CUDA_CHECK_ERROR(cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
	}

	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.DPs_out, size));

	size = KangCnt * 96;
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.Kangs, size));

	total_mem += JMP_CNT * 96;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96));

	total_mem += JMP_CNT * 96;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96));

	total_mem += JMP_CNT * 96;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96));

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.JumpsList, size));

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.DPTable, size));

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.L1S2, size));

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.LastPnts, size));

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.LoopTable, size));

	total_mem += 1024;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.dbg_buf, 1024));

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
        CUDA_CHECK_ERROR(cudaMalloc((void**)&Kparams.LoopedKangs, size));

        size = MAX_DP_CNT * GPU_DP_SIZE + 16;
        total_mem += size;
        err = buf_DPs_out.allocate(size);
        Kparams.DPs_out = buf_DPs_out.get<u32>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        size = KangCnt * 96;
        total_mem += size;
        err = buf_Kangs.allocate(size);
        Kparams.Kangs = buf_Kangs.get<u64>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        total_mem += JMP_CNT * 96;
        err = buf_Jumps1.allocate(JMP_CNT * 96);
        Kparams.Jumps1 = buf_Jumps1.get<u64>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        total_mem += JMP_CNT * 96;
        err = buf_Jumps2.allocate(JMP_CNT * 96);
        Kparams.Jumps2 = buf_Jumps2.get<u64>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        total_mem += JMP_CNT * 96;
        err = buf_Jumps3.allocate(JMP_CNT * 96);
        Kparams.Jumps3 = buf_Jumps3.get<u64>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        size = 2 * (u64)KangCnt * STEP_CNT;
        total_mem += size;
        err = buf_JumpsList.allocate(size);
        Kparams.JumpsList = buf_JumpsList.get<u64>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
        total_mem += size;
        err = buf_DPTable.allocate(size);
        Kparams.DPTable = buf_DPTable.get<u32>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        size = mpCnt * Kparams.BlockSize * sizeof(u64);
        total_mem += size;
        err = buf_L1S2.allocate(size);
        Kparams.L1S2 = buf_L1S2.get<u32>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        size = (u64)KangCnt * MD_LEN * (2 * 32);
        total_mem += size;
        err = buf_LastPnts.allocate(size);
        Kparams.LastPnts = buf_LastPnts.get<u64>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        size = (u64)KangCnt * MD_LEN * sizeof(u64);
        total_mem += size;
        err = buf_LoopTable.allocate(size);
        Kparams.LoopTable = buf_LoopTable.get<u64>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        total_mem += 1024;
        err = buf_dbg_buf.allocate(1024);
        Kparams.dbg_buf = buf_dbg_buf.get<u32>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }

        size = sizeof(u32) * KangCnt + 8;
        total_mem += size;
        err = buf_LoopedKangs.allocate(size);
        Kparams.LoopedKangs = buf_LoopedKangs.get<u32>();
        if (err != cudaSuccess)
        {
                printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
                return false;
        }


	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
        CUDA_CHECK_ERROR(cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice));
	free(buf);
//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
        CUDA_CHECK_ERROR(cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice));
	free(buf);

        CUDA_CHECK_ERROR(cuSetGpuParams(Kparams, jmp2_table));
        free(jmp2_table);
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
        CUDA_CHECK_ERROR(cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice));
	free(buf);

	printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
	return true;
}

void RCGpuKang::Release()
{
        free(RndPnts);
        free(DPs_out);
        buf_LoopedKangs.reset();  Kparams.LoopedKangs = nullptr;
        buf_dbg_buf.reset();      Kparams.dbg_buf = nullptr;
        buf_LoopTable.reset();    Kparams.LoopTable = nullptr;
        buf_LastPnts.reset();     Kparams.LastPnts = nullptr;
        buf_L1S2.reset();         Kparams.L1S2 = nullptr;
        buf_DPTable.reset();      Kparams.DPTable = nullptr;
        buf_JumpsList.reset();    Kparams.JumpsList = nullptr;
        buf_Jumps3.reset();       Kparams.Jumps3 = nullptr;
        buf_Jumps2.reset();       Kparams.Jumps2 = nullptr;
        buf_Jumps1.reset();       Kparams.Jumps1 = nullptr;
        buf_Kangs.reset();        Kparams.Kangs = nullptr;
        buf_DPs_out.reset();      Kparams.DPs_out = nullptr;
        if (!IsOldGpu) { buf_L2.reset(); Kparams.L2 = nullptr; }
}

void RCGpuKang::Stop()
{
	StopFlag = true;
}

void RCGpuKang::GenerateRndDistances()
{
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		if (i < KangCnt / 3)
			d.RndBits(Range - 4); //TAME kangs
		else
		{
			d.RndBits(Range - 1);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

bool RCGpuKang::Start()
{
	if (Failed)
		return false;

        
        CUDA_CHECK_ERROR(cudaSetDevice(CudaIndex));

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
	PntB = PntA;
	PntB.y.NegModP();

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	GenerateRndDistances();
/* 
	//we can calc start points on CPU
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);
		d.data[3] = 0;
		d.data[4] = 0;
		EcPoint p = ec.MultiplyG(d);
		memcpy(RndPnts[i].x, p.x.data, 32);
		memcpy(RndPnts[i].y, p.y.data, 32);
	}
	for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntA);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntB);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	//copy to gpu
        CUDA_CHECK_ERROR(cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice));
/**/
	//but it's faster to calc then on GPU
	u8 buf_PntA[64], buf_PntB[64];
	PntA.SaveToBuffer64(buf_PntA);
	PntB.SaveToBuffer64(buf_PntB);
	for (int i = 0; i < KangCnt; i++)
	{
		if (i < KangCnt / 3)
			memset(RndPnts[i].x, 0, 64);
		else
			if (i < 2 * KangCnt / 3)
				memcpy(RndPnts[i].x, buf_PntA, 64);
			else
				memcpy(RndPnts[i].x, buf_PntB, 64);
	}
	//copy to gpu
        CUDA_CHECK_ERROR(cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice));
	CallGpuKernelGen(Kparams);

        CUDA_CHECK_ERROR(cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8));
        cudaMemset(Kparams.dbg_buf, 0, 1024);
        cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
	int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
	u64* kangs = (u64*)malloc(kang_size);
        CUDA_CHECK_ERROR(cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost));
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &kangs[i * 12 + 8], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			if (i < 2 * KangCnt / 3)
				p = ec.AddPoints(PntA, p);
			else
				p = ec.AddPoints(PntB, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	return res;
}
#endif

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
        CUDA_CHECK_ERROR(cudaSetDevice(CudaIndex));

	if (!Start())
	{
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
        while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
                CUDA_CHECK_ERROR(cudaMemset(Kparams.DPs_out, 0, 4));
                CUDA_CHECK_ERROR(cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32)));
                CUDA_CHECK_ERROR(cudaMemset(Kparams.LoopedKangs, 0, 8));
		CallGpuKernelABC(Kparams);
		int cnt;
                CUDA_CHECK_ERROR(cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost));
		
		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
                        CUDA_CHECK_ERROR(cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost));
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

		//dbg
                CUDA_CHECK_ERROR(cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost));

                u32 lcnt;
                CUDA_CHECK_ERROR(cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost));
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
		}
		iter++;
#endif
	}

	Release();
}

int RCGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}