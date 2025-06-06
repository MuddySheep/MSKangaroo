// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "Ec.h"
#include "GpuBuffer.h" // why: RAII wrapper for GPU memory

#define STATS_WND_SIZE	16

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class RCGpuKang
{
private:
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	u32* DPs_out;
        TKparams Kparams; // raw pointers for kernels
        // RAII buffers to manage GPU memory
        GpuBuffer buf_L2;
        GpuBuffer buf_DPs_out;
        GpuBuffer buf_Kangs;
        GpuBuffer buf_Jumps1;
        GpuBuffer buf_Jumps2;
        GpuBuffer buf_Jumps3;
        GpuBuffer buf_JumpsList;
        GpuBuffer buf_DPTable;
        GpuBuffer buf_L1S2;
        GpuBuffer buf_LastPnts;
        GpuBuffer buf_LoopTable;
        GpuBuffer buf_dbg_buf;
        GpuBuffer buf_LoopedKangs;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntA;
	EcPoint PntB;

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];

	void GenerateRndDistances();
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif
public:
	int persistingL2CacheMaxSize;
	int CudaIndex; //gpu index in cuda
	int mpCnt;
	int KangCnt;
	bool Failed;
	bool IsOldGpu;

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();

	u32 dbg[256];

	int GetStatsSpeed();
};
