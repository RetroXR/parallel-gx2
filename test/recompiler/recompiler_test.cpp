// The PPC recompiler (src/Cafe/HW/Espresso/Recompiler: PPC to IML, the IML
// passes and register allocator, the x86-64 backend), checked against the
// interpreter: random runs of integer, logical, rotate, compare, condition
// register, load/store and branch instructions are compiled on the
// recompiler's own thread as a title's are, then run both ways from the same
// registers and memory, and every register, the condition register, XER's
// carry and summary overflow, CTR and the memory they wrote must agree,
// lwarx/stwcx. reservations among them. Only
// what the recompiler is known to turn down may be left to the interpreter.
// It also counts what the compile of each function allocates.
#include "Cafe/HW/Espresso/PPCState.h"
#include "Cafe/HW/Espresso/Recompiler/PPCRecompiler.h"
#include "Cafe/HW/Espresso/Recompiler/IML/IMLRegIdMap.h"
#include "Cafe/HW/MMU/MMU.h"
#include "util/MemMapper/MemMapper.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <new>
#include <random>
#include <thread>
#include <vector>

void PPCRecompiler_visitAddressNoBlock(uint32 enterAddress);
void PPCRecompiler_enter(PPCInterpreter_t* hCPU, PPCREC_JUMP_ENTRY funcPtr);
void PPCInterpreterSlim_executeInstruction(PPCInterpreter_t* hCPU);

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #x); failures++; } } while (0)

// every allocation the process makes, counted while main waits on a compile
static std::atomic<uint64> s_allocations{0};
void* operator new(size_t n)
{
	s_allocations.fetch_add(1, std::memory_order_relaxed);
	if (void* p = std::malloc(n ? n : 1))
		return p;
	throw std::bad_alloc();
}
void* operator new[](size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

static constexpr uint32 kCodeBase = 0x02000000;
static constexpr uint32 kCodeSize = 0x00400000;
static constexpr uint32 kDataBase = 0x10000000;
static constexpr uint32 kDataSize = 0x00010000;
static constexpr uint32 kFuncStride = 0x400; // room for 255 instructions

static void Put(uint32 address, uint32 v)
{
	*(uint32*)(memory_base + address) = _swapEndianU32(v);
}

// encodings
static uint32 D(uint32 op, uint32 rt, uint32 ra, uint32 imm) { return (op << 26) | (rt << 21) | (ra << 16) | (imm & 0xFFFF); }
static uint32 X(uint32 rs, uint32 ra, uint32 rb, uint32 xo, uint32 rc = 0) { return (31u << 26) | (rs << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc; }
static uint32 XO(uint32 rt, uint32 ra, uint32 rb, uint32 xo, uint32 rc = 0) { return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc; }
static uint32 RLW(uint32 op, uint32 rs, uint32 ra, uint32 sh, uint32 mb, uint32 me, uint32 rc) { return (op << 26) | (rs << 21) | (ra << 16) | (sh << 11) | (mb << 6) | (me << 1) | rc; }
static uint32 CMP(uint32 crf, uint32 ra, uint32 rb, bool logical) { return (31u << 26) | (crf << 23) | (ra << 16) | (rb << 11) | ((logical ? 32u : 0u) << 1); }
static uint32 CMPI(uint32 crf, uint32 ra, uint32 imm, bool logical) { return ((logical ? 10u : 11u) << 26) | (crf << 23) | (ra << 16) | (imm & 0xFFFF); }
static uint32 XL(uint32 bt, uint32 ba, uint32 bb, uint32 xo) { return (19u << 26) | (bt << 21) | (ba << 16) | (bb << 11) | (xo << 1); }
static uint32 BC(uint32 bo, uint32 bi, sint32 disp) { return (16u << 26) | (bo << 21) | (bi << 16) | ((uint32)disp & 0xFFFC); }
static constexpr uint32 kBlr = 0x4E800020;

static constexpr uint32 kBase = 30; // the data area's address, never written

class Gen
{
public:
	explicit Gen(uint32 seed, int kind = -1) : m_rng(seed), m_kind(kind) {}

	uint32 Reg() { return 3 + m_rng() % 10; } // r3..r12
	uint32 Imm() { return m_rng() % 4 == 0 ? (m_rng() & 0xFFFF) : (m_rng() % 64) - 32; }
	uint32 Rc() { return m_rng() % 3 == 0; }

	void Emit(std::vector<uint32>& code)
	{
		const uint32 rt = Reg(), ra = Reg(), rb = Reg();
		switch (m_kind >= 0 ? (uint32)m_kind : m_rng() % kKinds)
		{
		case 0: code.push_back(D(14, rt, ra, Imm())); break;                // addi
		case 1: code.push_back(D(15, rt, ra, Imm())); break;                // addis
		case 2: code.push_back(D(12, rt, ra, Imm())); break;                // addic
		case 3: code.push_back(D(13, rt, ra, Imm())); break;                // addic.
		case 4: code.push_back(XO(rt, ra, rb, 266, Rc())); break;           // add
		case 5: code.push_back(XO(rt, ra, rb, 10, Rc())); break;            // addc
		case 6: code.push_back(XO(rt, ra, rb, 138, Rc())); break;           // adde
		case 7: code.push_back(XO(rt, ra, 0, 202, Rc())); break;            // addze
		case 8: code.push_back(XO(rt, ra, 0, 234, Rc())); break;            // addme
		case 9: code.push_back(XO(rt, ra, rb, 40, Rc())); break;            // subf
		case 10: code.push_back(XO(rt, ra, rb, 8, Rc())); break;            // subfc
		case 11: code.push_back(XO(rt, ra, rb, 136, Rc())); break;          // subfe
		case 12: code.push_back(D(8, rt, ra, Imm())); break;                // subfic
		case 13: code.push_back(XO(rt, ra, 0, 104, Rc())); break;           // neg
		case 14: code.push_back(XO(rt, ra, rb, 235, Rc())); break;          // mullw
		case 15: code.push_back(D(7, rt, ra, Imm())); break;                // mulli
		case 16: code.push_back(XO(rt, ra, rb, 75, Rc())); break;           // mulhw
		case 17: code.push_back(XO(rt, ra, rb, 11, Rc())); break;           // mulhwu
		case 18: code.push_back(X(rt, ra, rb, 28, Rc())); break;            // and
		case 19: code.push_back(X(rt, ra, rb, 60, Rc())); break;            // andc
		case 20: code.push_back(X(rt, ra, rb, 444, Rc())); break;           // or
		case 21: code.push_back(X(rt, ra, rb, 412, Rc())); break;           // orc
		case 22: code.push_back(X(rt, ra, rb, 316, Rc())); break;           // xor
		case 23: code.push_back(X(rt, ra, rb, 124, Rc())); break;           // nor
		case 24: code.push_back(X(rt, ra, rb, 476, Rc())); break;           // nand
		case 25: code.push_back(X(rt, ra, rb, 284, Rc())); break;           // eqv
		case 26: code.push_back(D(28, rt, ra, m_rng())); break;             // andi.
		case 27: code.push_back(D(29, rt, ra, m_rng())); break;             // andis.
		case 28: code.push_back(D(24, rt, ra, m_rng())); break;             // ori
		case 29: code.push_back(D(25, rt, ra, m_rng())); break;             // oris
		case 30: code.push_back(D(26, rt, ra, m_rng())); break;             // xori
		case 31: code.push_back(X(rt, ra, rb, 24, Rc())); break;            // slw
		case 32: code.push_back(X(rt, ra, rb, 536, Rc())); break;           // srw
		case 33: code.push_back(X(rt, ra, rb, 792, Rc())); break;           // sraw
		case 34: code.push_back(X(rt, ra, m_rng() % 32, 824, Rc())); break; // srawi
		case 35: code.push_back(RLW(21, rt, ra, m_rng() % 32, m_rng() % 32, m_rng() % 32, Rc())); break; // rlwinm
		case 36: code.push_back(RLW(20, rt, ra, m_rng() % 32, m_rng() % 32, m_rng() % 32, Rc())); break; // rlwimi
		case 37: code.push_back(RLW(23, rt, ra, rb, m_rng() % 32, m_rng() % 32, Rc())); break;           // rlwnm
		case 38: code.push_back(X(rt, ra, 0, 26, Rc())); break;             // cntlzw
		case 39: code.push_back(X(rt, ra, 0, (m_rng() & 1) ? 954 : 922, Rc())); break; // extsb, extsh
		case 40: code.push_back(m_rng() & 1 ? CMP(m_rng() % 8, ra, rb, m_rng() & 1) : CMPI(m_rng() % 8, ra, Imm(), m_rng() & 1)); break;
		case 41:
		{
			static const uint32 crops[] = {257, 449, 193, 225, 33, 289, 129, 417}; // crand cror crxor crnand crnor creqv crandc crorc
			code.push_back(XL(m_rng() % 32, m_rng() % 32, m_rng() % 32, crops[m_rng() % 8]));
			break;
		}
		case 42: code.push_back(X(rt, 0, 0, 19)); break;                    // mfcr
		case 43:
		{
			// a load or store at an aligned offset in the data area
			static const uint32 ops[] = {32, 36, 40, 42, 44, 34, 38}; // lwz stw lhz lha sth lbz stb
			const uint32 op = ops[m_rng() % 7];
			const uint32 size = (op == 32 || op == 36) ? 4 : (op == 34 || op == 38) ? 1 : 2;
			const uint32 off = (m_rng() % (kDataSize / 2)) & ~(size - 1);
			code.push_back(D(op, rt, kBase, off));
			break;
		}
		case 44:
		{
			// a conditional skip over the next instruction
			const uint32 bo = (m_rng() & 1) ? 12 : 4;
			code.push_back(BC(bo, m_rng() % 32, 8));
			code.push_back(D(14, rt, ra, Imm()));
			break;
		}
		case 45:
		{
			// a reservation: lwarx, a change, stwcx. (and now and then a
			// stwcx. with none held), which sets cr0 from it and XER's SO
			const uint32 off = (m_rng() % (kDataSize / 2)) & ~3u;
			code.push_back(D(14, 0, 0, off));                           // li r0, off
			if (m_rng() % 4)
			{
				code.push_back(X(rt, kBase, 0, 20));                    // lwarx rt, r30, r0
				code.push_back(D(14, rt, rt, Imm()));                   // addi rt, rt, imm
			}
			code.push_back(X(rt, kBase, 0, 150, 1));                    // stwcx. rt, r30, r0
			break;
		}
		default:
		{
			// a small counted loop: mtctr, add, bdnz back
			code.push_back(D(14, 0, 0, 1 + m_rng() % 5));              // li r0, n
			code.push_back(X(0, 9, 0, 467));                            // mtctr r0 (spr 9)
			code.push_back(XO(rt, rt, ra, 266));                        // add rt, rt, ra
			code.push_back(BC(16, 0, -4));                              // bdnz -4
			break;
		}
		}
	}

	static constexpr uint32 kKinds = 47;

private:
	std::mt19937 m_rng;
	int m_kind;
};

static uint8 s_dataInit[kDataSize];

struct State
{
	PPCInterpreter_t cpu;
	std::vector<uint8> data;
};

static void Setup(PPCInterpreter_t& cpu, uint32 entry, uint32 seed)
{
	std::mt19937 rng(seed * 7919u + 1);
	std::memset(&cpu, 0, sizeof(cpu));
	cpu.instructionPointer = entry;
	for (int i = 0; i < 32; i++)
		cpu.gpr[i] = rng();
	cpu.gpr[kBase] = kDataBase;
	for (int i = 0; i < 32; i++)
		cpu.cr[i] = rng() & 1;
	cpu.xer_ca = rng() & 1;
	cpu.xer_so = rng() & 1;
	cpu.spr.LR = 0; // blr leaves through entry 0: back to the host
	cpu.spr.CTR = rng();
	cpu.remainingCycles = 1 << 28;
	cpu.LSQE = 1;
	cpu.PSE = 1;
	std::memcpy(memory_base + kDataBase, s_dataInit, kDataSize);
}

static void Compare(const PPCInterpreter_t& a, const std::vector<uint8>& aData, const PPCInterpreter_t& b, const std::vector<uint8>& bData, uint32 seed, bool& ok)
{
	auto bad = [&](const char* what, int i, uint32 x, uint32 y) {
		if (ok)
			std::fprintf(stderr, "seed %u: %s%d interpreter %08x recompiler %08x\n", seed, what, i, x, y);
		ok = false;
	};
	for (int i = 0; i < 32; i++)
		if (a.gpr[i] != b.gpr[i])
			bad("r", i, a.gpr[i], b.gpr[i]);
	for (int i = 0; i < 32; i++)
		if (a.cr[i] != b.cr[i])
			bad("cr bit ", i, a.cr[i], b.cr[i]);
	if (a.xer_ca != b.xer_ca)
		bad("xer ca", 0, a.xer_ca, b.xer_ca);
	if (a.xer_so != b.xer_so)
		bad("xer so", 0, a.xer_so, b.xer_so);
	if (a.spr.CTR != b.spr.CTR)
		bad("ctr", 0, a.spr.CTR, b.spr.CTR);
	if (a.instructionPointer != b.instructionPointer)
		bad("ip", 0, a.instructionPointer, b.instructionPointer);
	if (aData != bData)
		bad("data", 0, 0, 0);
}

// what the recompiler leaves to the interpreter: srawi by 0 and the
// condition register's logical operations
static bool MayBeRefused(const std::vector<uint32>& code)
{
	for (uint32 op : code)
	{
		const uint32 primary = op >> 26, xo = (op >> 1) & 0x3FF;
		if (primary == 31 && xo == 824 && ((op >> 11) & 31) == 0)
			return true;
		if (primary == 19 && (xo == 257 || xo == 449 || xo == 193 || xo == 225 || xo == 33 || xo == 289 || xo == 129 || xo == 417))
			return true;
	}
	return false;
}

struct RunStats
{
	uint32 compiled = 0;
	uint32 refused = 0;
	uint64 allocationsTotal = 0;
	uint64 allocationsMax = 0;
};

static uint32 s_nextFunction = 0;
static PPCREC_JUMP_ENTRY s_lastCode; // the last function compiled

// one function: compiled on the recompiler's thread, then run both ways;
// false when it was not compiled or the two disagree
static bool RunOne(const std::vector<uint32>& body, uint32 seed, RunStats& stats, const char* what)
{
	const uint32 entry = kCodeBase + (s_nextFunction++) * kFuncStride;
	std::vector<uint32> code = body;
	code.push_back(kBlr);
	for (size_t i = 0; i < code.size(); i++)
		Put(entry + (uint32)(i * 4), code[i]);

	// and a one-instruction function after it: the recompiler's one thread
	// takes them in order, so once that one is compiled this one is settled,
	// compiled or turned down
	const uint32 marker = entry + kFuncStride - 8;
	Put(marker, D(14, 3, 0, 1));
	Put(marker + 4, kBlr);

	const uint64 before = s_allocations.load();
	PPCRecompiler_visitAddressNoBlock(entry);
	PPCRecompiler_visitAddressNoBlock(marker);
	const auto start = std::chrono::steady_clock::now();
	auto jump = [](uint32 address) {
		return __atomic_load_n(&ppcRecompilerInstanceData->ppcRecompilerDirectJumpTable[address / 4], __ATOMIC_ACQUIRE);
	};
	auto settled = [](PPCREC_JUMP_ENTRY f) {
		return f != PPCRecompiler_leaveRecompilerCode_unvisited && f != PPCRecompiler_leaveRecompilerCode_visited;
	};
	while (!settled(jump(marker)) && std::chrono::steady_clock::now() - start < std::chrono::seconds(10))
		std::this_thread::sleep_for(std::chrono::microseconds(50));
	CHECK(settled(jump(marker)));
	const PPCREC_JUMP_ENTRY fn = jump(entry);
	const uint64 used = s_allocations.load() - before;
	if (fn == PPCRecompiler_leaveRecompilerCode_unvisited || fn == PPCRecompiler_leaveRecompilerCode_visited)
	{
		// left to the interpreter, as a title's is: right only for what the
		// recompiler turns down
		stats.refused++;
		if (MayBeRefused(code))
			return true;
		std::fprintf(stderr, "%s seed %u: not compiled:", what, seed);
		for (uint32 op : code)
			std::fprintf(stderr, " %08x", op);
		std::fprintf(stderr, "\n");
		return false;
	}
	stats.compiled++;
	s_lastCode = fn;
	stats.allocationsTotal += used;
	stats.allocationsMax = std::max(stats.allocationsMax, used);

	static State interp, rec;
	Setup(interp.cpu, entry, seed);
	for (int steps = 0; interp.cpu.instructionPointer != 0 && steps < 100000; steps++)
		PPCInterpreterSlim_executeInstruction(&interp.cpu);
	interp.data.assign(memory_base + kDataBase, memory_base + kDataBase + kDataSize);

	Setup(rec.cpu, entry, seed);
	PPCRecompiler_enter(&rec.cpu, fn);
	rec.data.assign(memory_base + kDataBase, memory_base + kDataBase + kDataSize);

	bool ok = true;
	Compare(interp.cpu, interp.data, rec.cpu, rec.data, seed, ok);
	if (!ok)
	{
		std::fprintf(stderr, "  %s:", what);
		for (uint32 op : code)
			std::fprintf(stderr, " %08x", op);
		std::fprintf(stderr, "\n");
	}
	return ok;
}

// The register-id set and map stand in for std::unordered_set and
// std::unordered_map in the register allocator: checked here against
// std::map under random use, with a value that owns memory (so a value not
// destroyed, or destroyed twice, shows under the sanitizers)
static void TestRegIdContainers()
{
	std::mt19937 rng(777);
	IMLRegIdSet set;
	IMLRegIdMap<std::string> map;
	std::map<IMLRegID, std::string> ref;
	for (int step = 0; step < 20000; step++)
	{
		const IMLRegID id = (IMLRegID)(rng() % IML_REG_ID_CAPACITY);
		const uint32 op = rng() % 8;
		if (op < 3)
		{
			const std::string v(40 + rng() % 40, (char)('a' + rng() % 26));
			const bool newSet = set.insert(id).second;
			const bool newMap = map.try_emplace(id, v).second;
			const bool newRef = ref.try_emplace(id, v).second;
			CHECK(newSet == newRef);
			CHECK(newMap == newRef);
		}
		else if (op == 3)
		{
			map[id] += "x";
			ref[id] += "x";
			set.insert(id);
		}
		else if (op < 6)
		{
			const size_t n = ref.erase(id);
			CHECK(set.erase(id) == n);
			CHECK(map.erase(id) == n);
		}
		else if (op == 6 && rng() % 64 == 0)
		{
			IMLRegIdMap<std::string> copy(map);
			map.clear();
			set.clear();
			CHECK(map.empty() && set.empty());
			map = copy;
			for (const auto& kv : copy)
				set.insert(kv.first);
		}
		CHECK(set.contains(id) == (ref.count(id) != 0));
		CHECK(map.count(id) == ref.count(id));
	}
	CHECK(set.size() == ref.size());
	CHECK(map.size() == ref.size());
	// both hand out the registers lowest first, the map with each value
	auto r = ref.begin();
	for (IMLRegID id : set)
	{
		CHECK(r != ref.end() && id == r->first);
		++r;
	}
	r = ref.begin();
	for (const auto& kv : map)
	{
		CHECK(r != ref.end() && kv.first == r->first && kv.second == r->second);
		++r;
	}
	CHECK(r == ref.end());
	IMLRegIdSet other;
	other.insert(IML_REG_ID_CAPACITY - 1);
	other.insert(0);
	set.insert_all(other);
	CHECK(set.contains(0) && set.contains(IML_REG_ID_CAPACITY - 1));
	CHECK(!set.contains(IML_REG_ID_CAPACITY));

	// the generator's guest-name to register map: every name found again,
	// however their hashes collide, and no more than a register id numbers
	auto names = std::make_unique<IMLNameRegMap>();
	std::vector<IMLName> added;
	// random names (a run of names, as the generator's are, spreads too
	// evenly to collide)
	while (added.size() < IML_REG_ID_CAPACITY)
	{
		const IMLName name = (IMLName)(rng() & 0x7FFFFFFF);
		if (names->find(name))
			continue;
		CHECK(names->add(name, IMLReg(IMLRegFormat::I32, IMLRegFormat::I32, 0, (IMLRegID)added.size())));
		added.push_back(name);
	}
	CHECK(!names->add(0xFFFFFFF0u, IMLReg(IMLRegFormat::I32, IMLRegFormat::I32, 0, 0)));
	CHECK(names->size() == IML_REG_ID_CAPACITY);
	for (size_t i = 0; i < added.size(); i++)
	{
		const IMLReg* reg = names->find(added[i]);
		CHECK(reg && reg->GetRegID() == (IMLRegID)i);
		CHECK(names->name(i) == added[i]);
	}
	CHECK(names->find(0xFFFFFFF0u) == nullptr);
}

int main()
{
	memory_base = (uint8*)MemMapper::ReserveMemory(nullptr, (size_t)0x100000000ull, MemMapper::PAGE_PERMISSION::P_RW);
	CHECK(memory_base != nullptr);
	// the null guard, as the core commits it: a run that leaves through LR 0
	// is offered to the recompiler at 0, which finds nothing to compile
	CHECK(MemMapper::AllocateMemory(memory_base, 0x10000, MemMapper::PAGE_PERMISSION::P_RW, true) != nullptr);
	CHECK(MemMapper::AllocateMemory(memory_base + kCodeBase, kCodeSize, MemMapper::PAGE_PERMISSION::P_RW, true) != nullptr);
	CHECK(MemMapper::AllocateMemory(memory_base + kDataBase, kDataSize + 0x1000, MemMapper::PAGE_PERMISSION::P_RW, true) != nullptr);
	TestRegIdContainers();
	std::mt19937 drng(12345);
	for (auto& b : s_dataInit)
		b = (uint8)drng();

	PPCRecompiler_init();
	PPCRecompiler_allocateRange(kCodeBase, kCodeSize);

	RunStats stats;
	// each kind of instruction alone first, so a failure names it
	for (uint32 kind = 0; kind < Gen::kKinds; kind++)
		for (uint32 n = 0; n < 8; n++)
		{
			std::vector<uint32> code;
			Gen gen(1000 + kind * 16 + n, (int)kind);
			while (code.size() < 6)
				gen.Emit(code);
			char what[32];
			std::snprintf(what, sizeof(what), "kind %u", kind);
			if (!RunOne(code, kind * 16 + n, stats, what))
				failures++;
		}
	// then runs of them mixed
	const uint32 kFunctions = 300;
	for (uint32 f = 0; f < kFunctions; f++)
	{
		std::vector<uint32> code;
		Gen gen(f + 1);
		const uint32 len = 4 + f % 60;
		while (code.size() < len)
			gen.Emit(code);
		if (!RunOne(code, f, stats, "mixed"))
			failures++;
	}
	std::printf("recompiler: %u functions compiled and run both ways, %u left to the interpreter; a compile (with its marker's) allocated %.1f times on average, %llu at most\n",
		stats.compiled, stats.refused, stats.compiled ? (double)stats.allocationsTotal / stats.compiled : 0.0, (unsigned long long)stats.allocationsMax);
	// A compile runs while a title plays: its register sets and maps are
	// fixed tables and its scratch and code buffer are reserved up front, so
	// it allocates under a hundred times rather than hundreds (722 on average and 3853 at most, when
	// each register id had a hash node)
	CHECK(stats.compiled > 0);
	CHECK(stats.allocationsTotal <= 92 * (uint64)stats.compiled);
	CHECK(stats.allocationsMax <= 400);
	// The next title: the recompiler shut down and started again. Its
	// functions were freed with it (LeakSanitizer sees to that), and its code
	// memory is filled again from the start - the same first function lands
	// where it did before, rather than in memory mapped anew.
	PPCRecompiler_Shutdown();
	PPCRecompiler_init();
	PPCRecompiler_allocateRange(kCodeBase, kCodeSize);
	{
		s_nextFunction = 0;
		RunStats again;
		std::vector<uint32> code;
		Gen gen(1000, 0);
		while (code.size() < 6)
			gen.Emit(code);
		CHECK(RunOne(code, 0, again, "again"));
		const PPCREC_JUMP_ENTRY second = s_lastCode;
		s_nextFunction = 0;
		PPCRecompiler_Shutdown();
		PPCRecompiler_init();
		PPCRecompiler_allocateRange(kCodeBase, kCodeSize);
		CHECK(RunOne(code, 0, again, "again"));
		CHECK(s_lastCode == second);
	}
	PPCRecompiler_Shutdown();
	if (failures)
	{
		std::fprintf(stderr, "recompiler_test: %d failure(s)\n", failures);
		return 1;
	}
	std::printf("recompiler: ok\n");
	return 0;
}
