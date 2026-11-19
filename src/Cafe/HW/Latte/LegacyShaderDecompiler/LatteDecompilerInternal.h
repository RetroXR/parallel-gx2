#pragma once

class StringBuf;

// a decompile's list of T, held in a per-thread pool that every list of T of the
// decompile appends to and LatteDecompiler_ResetScratch empties (its capacity
// kept): decompiling a shader while a game plays allocates no list storage once
// the pools have grown to the largest shader. A list is filled in one go, so its
// items lie together; one appended to after another list it is moved to the end
template<typename T>
class LatteScratchList
{
public:
	static std::vector<T>& Pool()
	{
		static thread_local std::vector<T> pool;
		return pool;
	}
	T* data() { return m_count ? Pool().data() + m_first : nullptr; }
	const T* data() const { return m_count ? Pool().data() + m_first : nullptr; }
	T* begin() { return data(); }
	T* end() { return data() + m_count; }
	const T* begin() const { return data(); }
	const T* end() const { return data() + m_count; }
	size_t size() const { return m_count; }
	bool empty() const { return m_count == 0; }
	T& operator[](size_t i) { return Pool()[m_first + i]; }
	const T& operator[](size_t i) const { return Pool()[m_first + i]; }
	T& back() { return Pool()[m_first + m_count - 1]; }
	T& emplace_back()
	{
		std::vector<T>& pool = Pool();
		if (m_count == 0)
			m_first = (uint32)pool.size();
		else if (m_first + m_count != pool.size())
		{
			// another list was appended since: this one moved to the end
			const uint32 from = m_first;
			m_first = (uint32)pool.size();
			for (uint32 k = 0; k < m_count; k++)
			{
				T item = pool[from + k];
				pool.push_back(item);
			}
		}
		m_count++;
		return pool.emplace_back();
	}
	T& emplace_back(const T& item)
	{
		T copy = item;
		return emplace_back() = copy;
	}
	void push_back(const T& item) { emplace_back(item); }
	std::reverse_iterator<T*> rbegin() { return std::reverse_iterator<T*>(end()); }
	std::reverse_iterator<T*> rend() { return std::reverse_iterator<T*>(begin()); }
	void clear() { m_count = 0; }
private:
	uint32 m_first{};
	uint32 m_count{};
};

struct LatteDecompilerALUInstruction
{
	struct LatteDecompilerCFInstruction* cfInstruction{};
	bool isOP3{};
	uint32 opcode{};
	uint32 instructionGroupIndex{};
	uint8 indexMode{};
	uint8 omod{};
	// destination
	uint32 destGpr{};
	uint8 destRel{};
	uint8 destElem{};
	uint8 destClamp{};
	uint8 writeMask{};
	// flags
	uint8 updateExecuteMask{};
	uint8 updatePredicate{};
	// source operands
	struct
	{
		uint32 sel{};
		uint8 rel{};
		uint8 abs{};
		uint8 neg{};
		uint8 chan{};
	}sourceOperand[3];
	union
	{
		uint32 w[4];
		float f[4];
	}literalData;
	// information from analyzer stage
	uint8 aluUnit{}; // 0-3 -> ALU.x/y/u/w (PV), 4 -> Trans unit (PS)
	uint8 indexInGroup{}; // index of instruction within instruction group
	bool isLastInstructionOfGroup{};
};

struct LatteDecompilerTEXInstruction
{
	struct LatteDecompilerCFInstruction* cfInstruction{};
	uint32 opcode{};
	// texture or vertex fetch
	// shared
	sint32 srcGpr;
	sint32 dstGpr;
	sint8 dstSel[4];
	// texture fetch
	struct
	{
		sint32 textureIndex{};
		sint32 samplerIndex{};
		uint32 offset{};
		sint8 srcSel[4]{};
		sint8 offsetX{};
		sint8 offsetY{};
		sint8 offsetZ{};
		bool unnormalized[4]{}; // set if texture coordinates are in [0,dim] range instead of [0,1]
		sint8 lodBias{}; // divide by 16 to get actual value
	}textureFetch;
	// memRead
	struct
	{
		uint32 arrayBase{};
		sint8 srcSelX{};
		Latte::E_HWFMT format{};
		uint8 nfa{};
		uint8 isSigned{};
	}memRead;
};

struct LatteDecompilerCFInstruction
{
	uint32 type{};
	uint32 cfAddr{};
	// for clauses with instructions
	uint32 addr{};
	sint32 count{};
	// clause contains either ALU or TEX instructions
	LatteScratchList<LatteDecompilerALUInstruction> instructionsALU;
	LatteScratchList<LatteDecompilerTEXInstruction> instructionsTEX;
	// for clauses that access uniform buffers
	uint32 cBank0Index{};
	uint32 cBank1Index{};
	uint32 cBank0AddrBase{};
	uint32 cBank1AddrBase{};
	// for exports
	uint32 exportType{};
	uint8  exportComponentSel[4]{};
	uint32 exportBurstCount{};
	// for mem write
	uint32 memWriteArraySize{};
	uint8 memWriteCompMask{};
	uint8 memWriteElemSize{}; // 0-3
	// for exports and mem write
	uint32 exportArrayBase{};
	uint32 exportSourceGPR{};
	// misc
	uint32 cfCond{};
	uint32 popCount{};
	// information from analyzer stage
	bool modifiesPredicate{};
	bool modifiesActiveMask{};
	uint32 numPredInstructions{};
	sint32 activeStackDepth{}; // stack depth during the clause/CF instruction

	LatteDecompilerCFInstruction()
	{

	}

	~LatteDecompilerCFInstruction()
	{
		cemu_assert_debug(!(instructionsALU.size() != 0 && instructionsTEX.size() != 0)); // make sure we haven't accidentally added the wrong instruction type
	}

	LatteDecompilerCFInstruction(const LatteDecompilerCFInstruction& mE) = default;
	LatteDecompilerCFInstruction(LatteDecompilerCFInstruction&& mE) = default;

	LatteDecompilerCFInstruction& operator=(LatteDecompilerCFInstruction&& mE) = default;
};

struct LatteDecompilerSubroutineInfo
{
	uint32 cfAddr;
	LatteScratchList<LatteDecompilerCFInstruction> instructions;
};

// helper struct to track the highest accessed offset within a buffer
struct LatteDecompilerBufferAccessTracker
{
	bool hasStaticIndexAccess{false};
	bool hasDynamicIndexAccess{false};
	sint32 highestAccessDynamicIndex{0};
	sint32 highestAccessStaticIndex{0};

	// track access, index is the array index and not a byte offset
	void TrackAccess(sint32 index, bool isDynamicIndex)
	{
		if (isDynamicIndex)
		{
			hasDynamicIndexAccess = true;
			if (index > highestAccessDynamicIndex)
				highestAccessDynamicIndex = index;
		}
		else
		{
			hasStaticIndexAccess = true;
			if (index > highestAccessStaticIndex)
				highestAccessStaticIndex = index;
		}
	}

	sint32 DetermineSize(uint64 shaderBaseHash, sint32 maximumSize) const
	{
		// here we try to predict the accessed byte range so we dont have to upload the whole buffer
		// if no bound can be determined then return maximumSize
		// for some known shaders we use hand-tuned values instead of the maximumSize fallback value that those shaders would normally use
		if(shaderBaseHash == 0x8ff56afdf1a2f837) // XCX text rendering
			return 24;
		if(shaderBaseHash == 0x37b9100c1310d3bb) // BotW UI backdrops 1
			return 24;
		if(shaderBaseHash == 0xf7ba548c1fefe24a) // BotW UI backdrops 2
			return 30;

		sint32 highestAccessIndex = -1;
		if(hasStaticIndexAccess)
			highestAccessIndex = highestAccessStaticIndex;
		if(hasDynamicIndexAccess)
			return maximumSize; // dynamic index exists and no bound can be determined
		if (highestAccessIndex < 0)
			return 1; // no access at all? But avoid zero as a size
		return highestAccessIndex + 1;
	}

	bool HasAccess() const
	{
		return hasStaticIndexAccess || hasDynamicIndexAccess;
	}

	bool HasRelativeAccess() const
	{
		return hasDynamicIndexAccess;
	}
};

struct LatteDecompilerShaderContext
{
	LatteDecompilerOutput_t* output;
	LatteDecompilerShader* shader;
	LatteConst::ShaderType shaderType;
	const class LatteDecompilerOptions* options;
	uint32* contextRegisters; // deprecated
	struct LatteContextRegister* contextRegistersNew;
	uint64 shaderBaseHash;
	StringBuf* shaderSource; // the thread's scratch (LatteDecompiler_SourceScratch)
	LatteScratchList<LatteDecompilerCFInstruction> cfInstructions;
	// the uniforms the shader reads, gathered here and copied to the shader
	// once the analyzer has them all
	LatteScratchList<LatteDecompilerRemappedUniformEntry_t> remappedUniforms;
	// fetch shader (required for vertex shader)
	LatteFetchShader* fetchShader{};
	// geometry copy shader (only present when geometry shader is active)
	LatteParsedGSCopyShader* parsedGSCopyShader; // gsCopyShader once parsed
	LatteParsedGSCopyShader gsCopyShader;
	// state
	bool hasError;
	// type tracker
	struct
	{
		// data type tracker
		uint8 defaultDataType;
		bool genFloatReg; // if set, generate R*f register variables
		bool genIntReg; // if set, generate R*i register variables
		bool useArrayGPRs; // if set, an array is used to represent GPRs instead of individual variables
	}typeTracker;
	// analyzer
	struct
	{
		// general
		bool hasStreamoutEnable{}; // set if streamout is enabled
		bool hasLoops{}; // loop directives present in shader
		// vertex shader
		bool isPointsPrimitive{}; // set if current render primitive is points
		bool outputPointSize{}; // set if the current shader should output the point size
		std::bitset<256> inputAttributSemanticMask; // one set bit for every used semanticId - todo: there are only 128 bit available semantic locations? The MSB has special meaning?
		// uniforms
		LatteDecompilerBufferAccessTracker uniformRegisterAccessTracker;
		LatteDecompilerBufferAccessTracker uniformBufferAccessTracker[LATTE_NUM_MAX_UNIFORM_BUFFERS];
		// ssbo
		bool hasSSBORead; // shader has instructions that read from SSBO
		bool hasSSBOWrite; // shader has instructions that write to SSBO
		// textures
		std::bitset<LATTE_NUM_MAX_TEX_UNITS> texUnitUsesTexelCoordinates;
		bool hasCubeMapTexture; // set to true if a cubemap texture is used
		bool hasGradientLookup; // set to true if texture lookup with custom gradients is used
		// misc
		bool usesRelativeGPRRead; // set if indexed GPR reads are used
		bool usesRelativeGPRWrite; // set if indexed GPR writes are used
		uint8 gprUseMask[(LATTE_NUM_GPR + 7) / 8]; // 1 bit per GPR, set if GPR is read/written anywhere in the program (ignores GPR accesses with relative index)
		bool hasStreamoutWrite; // stream-out CF instructions are used
		bool hasRedcCUBE; // has cube reduction instruction
		bool hasFragCoordAccess{false}; // accesses gl_FragCoord
		bool modifiesPixelActiveState; // set if the active mask is changed anywhere in the shader (If false, we can skip active mask checks)
		bool usesIntegerValues; // set if the shader uses any kind of integer instruction or integer-based GPR/AR access
		sint32 activeStackMaxDepth; // maximum depth of pixel state stack
		// analyzer stage (vs)
		bool writesPointSize{};
		// streamout (vs and gs)
		bool useSSBOForStreamout{};
		// geometry shader
		uint32 numEmitVertex{}; // counts how often emit vertex instruction is found
	}analyzer;

	// set while generating code for subroutine
	bool isSubroutine;
	LatteDecompilerSubroutineInfo* subroutineInfo;

	// emitter
	bool hasUniformVarBlock;
	sint32 currentBindingPointVK{};
	sint32 currentBufferBindingPointMTL{};
	sint32 currentTextureBindingPointMTL{};
	struct ALUClauseTemporariesState* aluPVPSState{nullptr};
	
	// misc
	LatteScratchList<LatteDecompilerSubroutineInfo> list_subroutines;
};

void LatteDecompiler_analyze(LatteDecompilerShaderContext* shaderContext, LatteDecompilerShader* shader);
void LatteDecompiler_analyzeDataTypes(LatteDecompilerShaderContext* shaderContext);
void LatteDecompiler_emitGLSLShader(LatteDecompilerShaderContext* shaderContext, LatteDecompilerShader* shader);
// false, with nothing written, for a shader it does not cover yet
bool LatteDecompiler_emitSPIRVShader(LatteDecompilerShaderContext* shaderContext, LatteDecompilerShader* shader);
#ifdef ENABLE_METAL
void LatteDecompiler_emitMSLShader(LatteDecompilerShaderContext* shaderContext, LatteDecompilerShader* shader);
#endif

void LatteDecompiler_cleanup(LatteDecompilerShaderContext* shaderContext);

// helper functions

sint32 LatteDecompiler_getColorOutputIndexFromExportIndex(LatteDecompilerShaderContext* shaderContext, sint32 exportIndex);
// a pixel shader input replaced by point sprite coordinates (SPI_INTERP_CONTROL_0's
// PNT_SPRITE_ENA, the input's PT_SPRITE_TEX, points rasterized): each
// component's override, 0 (0.0), 1 (1.0), 2 (S), 3 (T, 1-T where TOP_1) or
// -1 (interpolated); false where the input is not replaced
bool LatteDecompiler_GetSpriteOverride(LatteDecompilerShaderContext* shaderContext, sint32 inputIndex, sint8 sel[4], bool* topOne);
