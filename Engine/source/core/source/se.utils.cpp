#include "se.utils.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>
#include <imgui.h>
#include <se.gfx.hpp>
#ifdef _WIN32
    #define PATH_MAX 4096
#elif defined(__linux__)
    #include <unistd.h>
    #include <limits.h>
#endif

namespace glfw_static {
static bool gGLFWInitialized = false;
static int	gGLFWWindowCount = 0;

auto GLFWErrorCallback(int error, const char* description) -> void {
  se::error("GLFW Error ({}): {}", error, description);
}
}

namespace se {
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ platform                                                                  ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ Cross platform API for utils.  											  							     ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#ifdef _WIN32
  #include <windows.h>
  #include <commdlg.h>

  auto Platform::string_cast(const std::string& utf8str) noexcept -> std::wstring {
    if (utf8str.empty()) return L"";
    int char_count = MultiByteToWideChar(CP_UTF8, 0,
                                         utf8str.data(), (int)utf8str.size(),
                                         nullptr, 0);
    std::wstring conv(char_count, 0);
    MultiByteToWideChar(CP_UTF8, 0,
                        utf8str.data(), (int)utf8str.size(),
                        &conv[0], char_count);
    return conv;
  }

  auto Platform::string_cast(const std::wstring& utf16str) noexcept -> std::string {
    if (utf16str.empty()) return "";
    int char_count = WideCharToMultiByte(CP_UTF8, 0,
                                         utf16str.data(), (int)utf16str.size(),
                                         nullptr, 0,
                                         nullptr, nullptr);
    std::string conv(char_count, 0);
    WideCharToMultiByte(CP_UTF8, 0,
                        utf16str.data(), (int)utf16str.size(),
                        &conv[0], char_count,
                        nullptr, nullptr);
    return conv;
  }
  
  auto Platform::open_file(std::string const& filter, std::string const& path) noexcept -> std::string {
    void* window_handle = se::gfx::GFXContext::device()->from_which_adapter()->
      from_which_context()->get_binded_window()->get_handle();
    std::string path_reorganize = std::filesystem::path(path).string();
    se::info("hello {}", path_reorganize);

    OPENFILENAMEA ofn;
    CHAR szFile[260] = { 0 };
    ZeroMemory(&ofn, sizeof(OPENFILENAME));
    ofn.lStructSize = sizeof(OPENFILENAME);
    ofn.hwndOwner = *((HWND*)window_handle);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter.c_str();
    ofn.nFilterIndex = 1;
    ofn.lpstrInitialDir = path_reorganize.c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn) == TRUE) {
      return ofn.lpstrFile;
    }
    return std::string();
  }
  
  auto Platform::save_file(std::string const& filter, std::string const& name) noexcept -> std::string {
    void* window_handle = se::gfx::GFXContext::device()->from_which_adapter()->
      from_which_context()->get_binded_window()->get_handle();
    std::string path_reorganize = std::filesystem::path(name).string();
    std::string dir = Filesys::get_parent_path(path_reorganize);
    std::string stem = Filesys::get_filename(path_reorganize);
    se::info("hello {}", path_reorganize);

    OPENFILENAMEA ofn;
    CHAR szFile[260] = { 0 };
    memcpy(szFile, stem.c_str(), stem.size() + 1);
    ZeroMemory(&ofn, sizeof(OPENFILENAME));
    ofn.lStructSize = sizeof(OPENFILENAME);
    ofn.hwndOwner = *((HWND*)window_handle);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter.c_str();
    ofn.nFilterIndex = 1;
    ofn.lpstrInitialDir = dir.c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetSaveFileNameA(&ofn) == TRUE) {
      return ofn.lpstrFile;
    }
    return std::string();
  }

#elif defined(__linux__)
  #include <iconv.h>
  #include <errno.h>
  #include <cstring>
  #include <stdexcept>
  #include <cstdio>

  auto Platform::string_cast(const std::string& utf8str) noexcept -> std::wstring {
    if (utf8str.empty()) return L"";

    iconv_t cd = iconv_open("WCHAR_T", "UTF-8");
    if (cd == (iconv_t)(-1)) return L"";

    size_t in_size = utf8str.size();
    size_t out_size = (in_size + 1) * sizeof(wchar_t);
    std::wstring result(out_size / sizeof(wchar_t), 0);

    char* in_buf = const_cast<char*>(utf8str.data());
    char* out_buf = reinterpret_cast<char*>(&result[0]);
    size_t out_left = out_size;

    if (iconv(cd, &in_buf, &in_size, &out_buf, &out_left) == (size_t)(-1)) {
      iconv_close(cd);
      return L"";
    }

    iconv_close(cd);
    size_t converted = (out_size - out_left) / sizeof(wchar_t);
    result.resize(converted);
    return result;
  }

  auto Platform::string_cast(const std::wstring& wstr) noexcept -> std::string {
    if (wstr.empty()) return "";

    iconv_t cd = iconv_open("UTF-8", "WCHAR_T");
    if (cd == (iconv_t)(-1)) return "";

    size_t in_size = wstr.size() * sizeof(wchar_t);
    size_t out_size = in_size * 2;
    std::string result(out_size, 0);

    char* in_buf = reinterpret_cast<char*>(const_cast<wchar_t*>(wstr.data()));
    char* out_buf = &result[0];
    size_t out_left = out_size;

    if (iconv(cd, &in_buf, &in_size, &out_buf, &out_left) == (size_t)(-1)) {
      iconv_close(cd);
      return "";
    }

    iconv_close(cd);
    result.resize(out_size - out_left);
    return result;
  }
  
  std::string exec_zenity_command(const std::string& cmd) {
      std::array<char, 256> buffer;
      std::string result;
      std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
      if (!pipe) return "";
      while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
          result += buffer.data();
      }
      // Remove trailing newline
      if (!result.empty() && result.back() == '\n') result.pop_back();
      return result;
  }

  auto Platform::open_file(std::string const& filter, std::string const& path) noexcept -> std::string {
      std::string cmd = "zenity --file-selection";
      if (!path.empty()) cmd += " --filename=\"" + path + "/\"";
      return exec_zenity_command(cmd);
  }

  auto Platform::save_file(std::string const& filter, std::string const& name) noexcept -> std::string {
      std::string cmd = "zenity --file-selection --save";
      if (!name.empty()) cmd += " --filename=\"" + name + "\"";
      return exec_zenity_command(cmd);
  }
#else

  #error Platform::string_cast is not implemented for this OS

#endif

  timer::timer() {
    m_startTimePoint = std::chrono::steady_clock::now();
    m_prevTimePoint = m_startTimePoint;
  }

  auto timer::update() noexcept -> void {
    auto const now = std::chrono::steady_clock::now();
    uint64_t const deltaTimeCount = uint64_t(
      std::chrono::duration<double, std::micro>(now - m_prevTimePoint).count());
    m_deltaTime = 0.000001 * deltaTimeCount;
    m_prevTimePoint = now;
  }

  auto timer::delta_time() noexcept -> double { return m_deltaTime; }

  auto timer::total_time() noexcept -> double {
    uint64_t const totalTimeCount = uint64_t(
      std::chrono::duration<double, std::micro>(m_prevTimePoint - m_startTimePoint).count());
    return 0.000001 * totalTimeCount;
  }
  
  auto Worldtime::get() noexcept -> Worldtime {
    Worldtime wtp;
    using namespace std;
    using namespace std::chrono;
    typedef duration<int, ratio_multiply<hours::period, ratio<24> >::type> days;
    system_clock::time_point now = system_clock::now();
    system_clock::duration tp = now.time_since_epoch();
    wtp.y = duration_cast<years>(tp);
    tp -= wtp.y;
    wtp.d = duration_cast<days>(tp);
    tp -= wtp.d;
    wtp.h = duration_cast<hours>(tp);
    tp -= wtp.h;
    wtp.m = duration_cast<minutes>(tp);
    tp -= wtp.m;
    wtp.s = duration_cast<seconds>(tp);
    tp -= wtp.s;
    return wtp;
  }

  auto Worldtime::to_string() noexcept -> std::string {
    std::string str;
    str += std::to_string(y.count());
    str += std::to_string(d.count());
    str += std::to_string(h.count());
    str += std::to_string(m.count());
    str += std::to_string(s.count());
    return str;
  }

  auto Configuration::set_macro(std::string const& name, std::string const& path) noexcept -> void {
    Singleton<Configuration>::instance()->m_macroDefined[name] = path;
  }

  auto Configuration::set_config_file(std::string const& path) noexcept -> void {
    Singleton<Configuration>::instance()->m_configFilePath = path;
    MiniBuffer condigdata;
    Filesys::sync_read_file(path, condigdata);
    YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(condigdata.m_data));
    // check scene name
    std::string engine_path = Filesys::get_absolute_path(Filesys::preprocess(data["engine_path"].as<std::string>()));
    Singleton<Configuration>::instance()->m_stringProperties["engine_path"] = engine_path;
    std::string project_path = Filesys::get_parent_path(Filesys::get_absolute_path(Filesys::preprocess(path)));
    Singleton<Configuration>::instance()->m_stringProperties["project_path"] = project_path;
    se::info("Configuration: engine_path found: {}", engine_path);
    se::info("Configuration: project_path found: {}", project_path);

    std::vector<std::string> shader_paths = { project_path };
    if (data["shader_path"]) {
      for (auto node : data["shader_path"]) {
        std::filesystem::path parsed_path = node.as<std::string>();
        if (parsed_path.is_absolute()) {
          shader_paths.push_back(node.as<std::string>());
        }
        else {
          std::filesystem::path engine_cat = std::filesystem::path(engine_path) / parsed_path;
          if (std::filesystem::exists(engine_cat)) {
            shader_paths.push_back(engine_cat.lexically_normal().string());
          }
          std::filesystem::path project_cat = std::filesystem::path(project_path) / parsed_path;
          if (std::filesystem::exists(project_cat)) {
            shader_paths.push_back(project_cat.lexically_normal().string());
          }
        }
      }
    }
    shader_paths.push_back(engine_path);
    Singleton<Configuration>::instance()->m_stringArrayProperties["shader_path"] = shader_paths;
  }

  auto Configuration::string_property(std::string const& name) noexcept -> std::string const& {
    static std::string str_unkown = "UNKOWN_STRING";
    auto find = Singleton<Configuration>::instance()->m_stringProperties.find(name);
    if (find == Singleton<Configuration>::instance()->m_stringProperties.end()) { return str_unkown; }
    else { return find->second; }
  }

  auto Configuration::string_array_property(std::string const& name) noexcept -> std::vector<std::string> const& {
    static std::vector<std::string> empty_array = {};
    auto find = Singleton<Configuration>::instance()->m_stringArrayProperties.find(name);
    if (find == Singleton<Configuration>::instance()->m_stringArrayProperties.end()) { return empty_array; }
    else { return find->second; }
  }
  
  void ShowStringMapGui(const std::unordered_map<std::string, std::string>& stringMap) {
    static char searchBuf[128] = "";
    ImGui::InputText("Search", searchBuf, IM_ARRAYSIZE(searchBuf));

    // Compute the default width of the left column based on the widest key
    static float keyColumnWidth = 0.0f;
    static bool initialized = false;
    if (!initialized) {
      float maxWidth = 0.0f;
      for (const auto& [key, _] : stringMap) {
        ImVec2 textSize = ImGui::CalcTextSize(key.c_str());
        if (textSize.x > maxWidth)
          maxWidth = textSize.x;
      }
      keyColumnWidth = maxWidth + 20.0f; // add padding
      initialized = true;
    }
    if (ImGui::BeginTable("StringMapTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
      ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, keyColumnWidth);
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      for (const auto& [key, value] : stringMap) {
        if (strlen(searchBuf) > 0 &&
          key.find(searchBuf) == std::string::npos &&
          value.find(searchBuf) == std::string::npos)
          continue;

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(key.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::TextUnformatted(value.c_str());
      }

      ImGui::EndTable();
    }
  }

  void ShowMultiMapGui(const std::unordered_map<std::string, std::vector<std::string>>& multimap) {
    static char searchBuf[128] = "";
    ImGui::InputText("Search", searchBuf, IM_ARRAYSIZE(searchBuf));

    // Compute the default width of the left column based on the widest key
    static float keyColumnWidth = 0.0f;
    static bool initialized = false;
    if (!initialized) {
      float maxWidth = 0.0f;
      for (const auto& [key, _] : multimap) {
        ImVec2 textSize = ImGui::CalcTextSize(key.c_str());
        if (textSize.x > maxWidth)
          maxWidth = textSize.x;
      }
      keyColumnWidth = maxWidth + 20.0f; // add padding
      initialized = true;
    }
    if (ImGui::BeginTable("MultiMapTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
      ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, keyColumnWidth);
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      for (const auto& [key, values] : multimap) {
        // Search filtering (matches key or any value)
        bool match = strlen(searchBuf) == 0 || key.find(searchBuf) != std::string::npos;
        for (const auto& val : values) {
          if (val.find(searchBuf) != std::string::npos) {
            match = true;
            break;
          }
        }
        if (!match) continue;

        if (values.empty()) {
          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted(key.c_str());
          ImGui::TableSetColumnIndex(1);
          ImGui::TextDisabled("(empty)");
        }
        else {
          for (size_t i = 0; i < values.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            if (i == 0)
              ImGui::TextUnformatted(key.c_str());
            else
              ImGui::TextUnformatted("");  // Empty cell for repeated key

            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(values[i].c_str());
          }
        }
      }

      ImGui::EndTable();
    }
  }

  auto Configuration::on_draw_gui() noexcept -> void {
    ImGui::Text("String properties");
    ImGui::PushID("m_stringProperties");
    ShowStringMapGui(Singleton<Configuration>::instance()->m_stringProperties);
    ImGui::PopID();
    ImGui::NewLine();
    ImGui::Text("String vector properties");
    ImGui::PushID("m_stringArrayProperties");
    ShowMultiMapGui(Singleton<Configuration>::instance()->m_stringArrayProperties);
    ImGui::PopID();
  }

  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ memory                                                                    ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ Memory management system.                              							     ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
namespace impl {

#ifndef ALIGN
#define ALIGN(x, a) (((x) + ((a)-1)) & ~((a)-1))
#endif

  struct BlockHeader {
    // union-ed with data
    BlockHeader* pNext;
  };

  struct PageHeader {
    // followed by blocks in this page
    PageHeader* pNext;
    // helper function that gives the first block
    auto blocks() noexcept -> BlockHeader* {
      return reinterpret_cast<BlockHeader*>(this + 1);
    }
  };

  struct Allocator {
    Allocator(); ~Allocator();
    Allocator(size_t data_size, size_t page_size, size_t alignment);
    // resets the allocator to a new configuration
    auto reset(size_t data_size, size_t page_size, size_t alignment) noexcept
      -> void;
    // alloc and free blocks
    auto allocate() noexcept -> void*;
    auto free(void* p) noexcept -> void;
    auto freeAll() noexcept -> void;
  private:
    // the page list
    PageHeader* pPageList = nullptr;
    // the free block list
    BlockHeader* pFreeList;
    // size definition
    size_t dataSize;
    size_t pageSize;
    size_t alignmentSize;
    size_t blockSize;
    uint32_t blocksPerPage;
    // statistics
    uint32_t numPages;
    uint32_t numBlocks;
    uint32_t numFreeBlocks;
    // gets the next block
    auto nextBlock(BlockHeader* pBlock) noexcept -> BlockHeader*;
    // disable copy & assignment
    Allocator(const Allocator& clone) = delete;
    auto operator=(const Allocator& rhs)->Allocator & = delete;
  };


  Allocator::Allocator() : pPageList(nullptr), pFreeList(nullptr) {}

  Allocator::Allocator(size_t data_size, size_t page_size, size_t alignment)
    : pPageList(nullptr), pFreeList(nullptr) {
    reset(data_size, page_size, alignment);
  }

  Allocator::~Allocator() { freeAll(); }

  auto Allocator::reset(size_t data_size, size_t page_size,
    size_t alignment) noexcept -> void {
    freeAll();

    dataSize      = data_size;
    pageSize      = page_size;
    alignmentSize = alignment;

    // Block must at least satisfy BlockHeader alignment
    size_t align = std::max(alignmentSize, alignof(BlockHeader));

    // Size of one block = header + data, rounded up to required alignment
    blockSize = ALIGN(sizeof(BlockHeader) + dataSize,
                std::max(alignmentSize, alignof(BlockHeader)));


    // Number of blocks that fit in a page *after* the PageHeader
    size_t usable = pageSize - sizeof(PageHeader);
    blocksPerPage = static_cast<uint32_t>(usable / blockSize);

    // Basic sanity checks to catch configuration errors early
    assert(blocksPerPage > 0 && "page too small for even one block");
    assert((blockSize % alignof(BlockHeader)) == 0);
  }

  auto Allocator::allocate() noexcept -> void* {
    if (!pFreeList) {
      // allocate a new page
      PageHeader* pNewPage = reinterpret_cast<PageHeader*>(new uint8_t[pageSize]);
      pNewPage->pNext = nullptr;

      ++numPages;
      numBlocks += blocksPerPage;
      numFreeBlocks += blocksPerPage;

      if (pPageList) {
        pNewPage->pNext = pPageList;
      }

      pPageList = pNewPage;

      BlockHeader* pBlock = pNewPage->blocks();
      // link each block in the page
      for (uint32_t i = 0; i < blocksPerPage - 1; i++) {
        pBlock->pNext = nextBlock(pBlock);
        pBlock = nextBlock(pBlock);
      }
      pBlock->pNext = nullptr;

      pFreeList = pNewPage->blocks();
    }

    BlockHeader* freeBlock = pFreeList;
    pFreeList = pFreeList->pNext;
    --numFreeBlocks;

    return reinterpret_cast<void*>(freeBlock);
  }

  auto Allocator::free(void* p) noexcept -> void {
    BlockHeader* block = reinterpret_cast<BlockHeader*>(p);
    block->pNext = pFreeList;
    pFreeList = block;
    ++numFreeBlocks;
  }

  auto Allocator::freeAll() noexcept -> void {
    PageHeader* pPage = pPageList;
    while (pPage) {
      PageHeader* _p = pPage;
      pPage = pPage->pNext;

      delete[] reinterpret_cast<uint8_t*>(_p);
    }

    pPageList = nullptr;
    pFreeList = nullptr;

    numPages = 0;
    numBlocks = 0;
    numFreeBlocks = 0;
  }

  auto Allocator::nextBlock(BlockHeader* pBlock) noexcept -> BlockHeader* {
    return reinterpret_cast<BlockHeader*>(reinterpret_cast<uint8_t*>(pBlock) +
      blockSize);
  }

  // size_t* Memory::pBlockSizeLookup;
  // Allocator* Memory::pAllocators;

  static const uint32_t kBlockSizes[] = {
    // 4-increments
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
    80, 84, 88, 92, 96,
    // 32-increments
    128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576,
    608, 640,
    // 64-increments
    704, 768, 832, 896, 960, 1024 };

  static const uint32_t kPageSize = 8192;
  static const uint32_t kAlignment = 4;

  // number of elements in the block size array
  static const uint32_t kNumBlockSizes =
    sizeof(kBlockSizes) / sizeof(kBlockSizes[0]);

  // largest valid block size
  static const uint32_t kMaxBlockSize = kBlockSizes[kNumBlockSizes - 1];

  struct MemoryManager {
    SINGLETON(MemoryManager, {
      // initialize block size lookup table
      pBlockSizeLookup = new size_t[kMaxBlockSize + 1];
      size_t j = 0;
      for (size_t i = 0; i <= kMaxBlockSize; i++) {
        if (i > kBlockSizes[j]) ++j;
          pBlockSizeLookup[i] = j;
      }
      // initialize the allocators
      pAllocators = new Allocator[kNumBlockSizes];
      for (size_t i = 0; i < kNumBlockSizes; i++) {
        pAllocators[i].reset(kBlockSizes[i], kPageSize, kAlignment);
      }
      });
    ~MemoryManager();
    auto allocate(size_t size) noexcept -> void*;
    auto free(void* p, size_t size) noexcept -> void;
  private:
    size_t* pBlockSizeLookup;
    Allocator* pAllocators;
    auto lookUpAllocator(size_t size) noexcept -> Allocator*;
  };

  MemoryManager::~MemoryManager() {

  }

#define L1_CACHE_LINE_SIZE 64

  inline auto AllocAligned(size_t size) -> void* {
    #ifdef _WIN32
      return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
    #elif defined(__linux__)
      void* ptr = nullptr;
      posix_memalign(&ptr, L1_CACHE_LINE_SIZE, size);
      return ptr;
    #endif
  }

  template <class T>
  inline auto AllocAligned(size_t count) -> T* {
    return (T*)AllocAligned(count * sizeof(T));
  }

  inline auto FreeAligned(void* p) -> void { 
    #ifdef _WIN32
      _aligned_free(p);
    #elif defined(__linux__)
      free(p);
    #endif
   }

  auto MemoryManager::allocate(size_t size) noexcept -> void* {
    Allocator* pAlloc = lookUpAllocator(size);
    if (pAlloc) return pAlloc->allocate();
    else return AllocAligned(size);
  }

  auto MemoryManager::free(void* p, size_t size) noexcept -> void {
    Allocator* pAlloc = lookUpAllocator(size);
    if (pAlloc) pAlloc->free(p);
    else FreeAligned(p);
  }

  auto MemoryManager::lookUpAllocator(size_t size) noexcept -> Allocator* {
    // check eligibility for lookup
    if (size <= kMaxBlockSize) return pAllocators + pBlockSizeLookup[size];
    else return nullptr;
  }

  }

  auto Memory::allocate(size_t size) noexcept -> void* {
    return Singleton<impl::MemoryManager>::instance()->allocate(size);
  }

  auto Memory::free(void* p, size_t size) noexcept -> void {
    Singleton<impl::MemoryManager>::instance()->free(p, size);
  }

  MiniBuffer::MiniBuffer() : m_data(nullptr), m_size(0), m_isReference(false) {}
  MiniBuffer::MiniBuffer(size_t size) : m_size(size), m_isReference(false) {
    m_data = Memory::allocate(size);
  }
  MiniBuffer::MiniBuffer(MiniBuffer const& b) {
    release(); m_size = b.m_size; m_data = Memory::allocate(m_size);
    m_isReference = false; memcpy(m_data, b.m_data, m_size);
  }
  MiniBuffer::MiniBuffer(MiniBuffer&& b) {
    release(); m_size = b.m_size; m_data = b.m_data;
    m_isReference = false; b.m_data = nullptr; b.m_size = 0;
  }
  MiniBuffer::MiniBuffer(void* data, size_t size)
    : m_data(data), m_size(size), m_isReference(true) {}
  MiniBuffer::~MiniBuffer() { release(); }
  auto MiniBuffer::operator=(MiniBuffer const& b) -> MiniBuffer& {
    release(); m_size = b.m_size; m_data = Memory::allocate(m_size);
    memcpy(m_data, b.m_data, m_size); return *this;
  }
  auto MiniBuffer::operator=(MiniBuffer&& b) -> MiniBuffer& {
    release(); m_size = b.m_size; m_data = b.m_data;
    b.m_data = nullptr; b.m_size = 0; return *this;
  }
  auto MiniBuffer::release() noexcept -> void {
    if (m_data == nullptr || m_isReference) return;
    Memory::free(m_data, m_size);
    m_data = nullptr; m_size = 0;
  }
  
  auto Filesys::preprocess(std::string const& path) noexcept -> std::string {
    std::string path_out = path;
    for (auto& iter : Singleton<Configuration>::instance()->m_macroDefined) {
      size_t pos = path_out.find("$" + iter.first + "$");
      if (pos != std::string::npos) {
        path_out.replace(pos, iter.first.length() + 2, iter.second);
      }
    }
    return path_out;
  }

  auto Filesys::get_executable_path() noexcept -> std::string {
    char buffer[PATH_MAX];
    std::string path;

#ifdef _WIN32
    DWORD len = GetModuleFileNameA(NULL, buffer, PATH_MAX);
    path = std::string(buffer, len);
#elif defined(__linux__)
    ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
    if (count == -1) {
        se::error("Platform :: linux :: Cannot read /proc/self/exe");
    }
    path = std::string(buffer, count);
#endif
    // Strip filename to get directory
    size_t last_slash = path.find_last_of("/\\");
    return path.substr(0, last_slash);
  }

  auto Filesys::get_absolute_path(std::string const& path) noexcept -> std::string {
    std::filesystem::path filepath = std::filesystem::path(path);
    if (filepath.is_relative()) {
      std::filesystem::path executable_path = std::filesystem::path(get_executable_path());
      return (executable_path / filepath).lexically_normal().string();
    }
    return filepath.lexically_normal().string();
  }
  
  auto Filesys::file_exists(std::string const& path) noexcept -> bool {
    std::filesystem::path file_path = std::filesystem::path(path);
    if (std::filesystem::exists(file_path)) return true;
    else return false;
  }
  
  auto Filesys::resolve_path(std::string const& path, std::vector<std::string> const& search_paths) noexcept -> std::string {
    std::filesystem::path input_path = std::filesystem::path(path);
    if (input_path.is_absolute()) {
      return input_path.lexically_normal().string();
    }

    for (auto const& search_path : search_paths) {
      std::filesystem::path prev_path = std::filesystem::path(search_path);
      std::filesystem::path concat = prev_path / input_path;
      if (std::filesystem::exists(concat)) {
        return concat.lexically_normal().string();
      }
    }

    se::error("Filsys :: resolve path {} not found. Searched following potential solution:", path);
    for (auto const& search_path : search_paths) {
      std::filesystem::path prev_path = std::filesystem::path(search_path);
      std::filesystem::path concat = prev_path / input_path;
      se::error("\t\t - {}", concat.lexically_normal().string());
    }

    return "./";
  }

  auto Filesys::get_parent_path(std::string const& path) noexcept -> std::string {
    return std::filesystem::path(path).parent_path().string();
  }

  auto Filesys::get_stem(std::string const& path) noexcept -> std::string {
    return std::filesystem::path(path).stem().string();
  }
  
  auto Filesys::get_extension(std::string const& path) noexcept -> std::string {
    return std::filesystem::path(path).extension().string();
  }

  auto Filesys::get_filename(std::string const& path) noexcept -> std::string {
    return std::filesystem::path(path).filename().string();
  }

  auto Filesys::sync_read_file(std::string const& path, MiniBuffer& buffer) noexcept -> bool {
    std::filesystem::path abs_path = std::filesystem::absolute(path);
    std::filesystem::path root = abs_path.root_path();
    std::ifstream ifs(path, std::ifstream::binary);
    if (ifs.is_open()) {
      ifs.seekg(0, std::ios::end);
      size_t size = size_t(ifs.tellg());
      buffer = se::MiniBuffer(size + 1);
      buffer.m_size = size;
      ifs.seekg(0);
      ifs.read(reinterpret_cast<char*>(buffer.m_data), size);
      ((char*)buffer.m_data)[size] = '\0';
      ifs.close();
    }
    else {
      se::error("Core.IO:SyncRW::syncReadFile() failed, file \'{}\' not found.", path);
    }
    return false;
  }
  
  auto Filesys::sync_write_file(std::string const& path, MiniBuffer& buffer) noexcept -> bool {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    if (ofs.is_open()) {
      ofs.write((char*)buffer.m_data, buffer.m_size);
      ofs.close();
    }
    else {
      se::error("Core.IO:SyncRW::syncWriteFile() failed, file \'{}\' open failed.", path);
    }
    return false;
  }
  
  auto Resources::query_runtime_uid() noexcept -> UID {
    static UID counter = 1'000'000'000;  // Avoid collision with file-based hashes
    return counter++;
  }

  auto Resources::query_string_uid(std::string const& str) noexcept -> UID {
    std::hash<std::string> hasher;
    return hasher(str);
  }
  
  auto Resources::query_string_uid(std::string_view str) noexcept -> UID {
    std::hash<std::string_view> hasher;
    return hasher(str);
  }


  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ window                                                                    ┃
  // ┠───────────────────────────────────────────────────────────────────────────┨
  // ┃ Input system, loaded from glfw.											  							     ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

	Window::Window(size_t width, size_t height, std::wstring const& name)
		: m_width(width), m_height(height), m_name(name) { init(); }

	auto Window::init() noexcept -> void {
    // if no instance of glfw is initialized
    if (!glfw_static::gGLFWInitialized) {
      if (!glfwInit()) error("GLFW :: Initialization failed");
      glfwSetErrorCallback(glfw_static::GLFWErrorCallback);
      glfw_static::gGLFWInitialized = true;
    }
    glfw_static::gGLFWWindowCount++;
    m_input = Input{ this };

    // Context hint selection
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_wndHandle = glfwCreateWindow(m_width, m_height, 
      Platform::string_cast(m_name).c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(m_wndHandle, this);
    // Set GLFW Callbacks
    glfwSetWindowSizeCallback(m_wndHandle, [](GLFWwindow* window, int width, int height) {
      Window* this_window = (Window*)glfwGetWindowUserPointer(window);
      this_window->m_onResizeSignal.emit(width, height);
    });
    glfwSetWindowIconifyCallback(m_wndHandle, [](GLFWwindow* window, int iconified) {
      Window* this_window = (Window*)glfwGetWindowUserPointer(window);
      this_window->m_iconified = (iconified == GLFW_TRUE);
    });
	}

  auto Window::is_running() noexcept -> bool {
    return !m_shouldQuit;
  }

  auto Window::fetch_events() noexcept -> int {
    if (glfwWindowShouldClose(m_wndHandle))
      m_shouldQuit = true;
    glfwPollEvents();
    return 0;
  }

  auto Window::destroy() noexcept -> void {
    glfwDestroyWindow(m_wndHandle);
    glfw_static::gGLFWWindowCount--;
    if (glfw_static::gGLFWWindowCount <= 0) {
      glfwTerminate();
      glfw_static::gGLFWInitialized = false;
    }
  }

  auto Window::resize(size_t x, size_t y) noexcept -> void {
    error("TODO :: Window_GLFW does not support func { resize(size_t x, size_t y) } for now!");
  }

  auto Window::get_high_dpi() noexcept -> float {
    float xscale, yscale;
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    glfwGetMonitorContentScale(primary, &xscale, &yscale);
    return xscale;
  }

  auto Window::connect_resize_event(std::function<void(size_t, size_t)> const& func) noexcept -> void {
    m_onResizeSignal.connect(func);
  }

  auto Window::get_framebuffer_size(int* width, int* height) noexcept -> void {
    glfwGetFramebufferSize(m_wndHandle, width, height);
  }

	auto Window::is_resized() noexcept -> bool {
    int width_new; int height_new;
    get_framebuffer_size(&width_new, &height_new);

    if (width_new != m_width || height_new != m_height) {
      m_width = width_new;
      m_height = height_new;
      return true;
    }
    return false;
  }

  auto Input::is_key_pressed(CodeEnum keycode) noexcept -> bool {
    auto window = static_cast<GLFWwindow*>(m_attachedWindow->get_handle());
    auto state = glfwGetKey(window, (int)keycode);
    return state == GLFW_PRESS || state == GLFW_REPEAT;
  }

  auto Input::get_mouse_x() noexcept -> float {
    auto window = static_cast<GLFWwindow*>(m_attachedWindow->get_handle());
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return (float)xpos;
  }

  auto Input::get_mouse_y() noexcept -> float {
    auto window = static_cast<GLFWwindow*>(m_attachedWindow->get_handle());
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return (float)ypos;
  }

  auto Input::enable_cursor() noexcept -> void {
    auto window = static_cast<GLFWwindow*>(m_attachedWindow->get_handle());
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  }

  auto Input::disable_cursor() noexcept -> void {
    auto window = static_cast<GLFWwindow*>(m_attachedWindow->get_handle());
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  }

  auto Input::get_mouse_position(int button) noexcept -> std::pair<float, float> {
    auto window = static_cast<GLFWwindow*>(m_attachedWindow->get_handle());
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return { (float)xpos, (float)ypos };
  }

  auto Input::get_mouse_scroll_x() noexcept -> float {
    return 0;
  }

  auto Input::get_mouse_scroll_y() noexcept -> float {
    return 0;
  }

  auto Input::is_mouse_button_pressed(CodeEnum button) noexcept -> bool {
    auto window = static_cast<GLFWwindow*>(m_attachedWindow->get_handle());
    auto state = glfwGetMouseButton(window, (int)button);
    return state == GLFW_PRESS;
  }

  auto ProfileSession::write_header() noexcept -> void {
    outputStream << "{\"otherData\": {},\"traceEvents\":[";
    outputStream.flush();
  }

  auto ProfileSession::write_footer() noexcept -> void {
    outputStream << "]}";
    outputStream.flush();
  }

  auto ProfileSession::write_segment(ProfileSegment const& seg) noexcept -> void {
    if (!outputStream.is_open()) return;
    if (profileCount++ > 0) outputStream << ",";
    std::string name = seg.tag;
    outputStream << "{";
    outputStream << "\"cat\":\"function\",";
    outputStream << "\"dur\":" << (seg.end - seg.start) << ',';
    outputStream << "\"name\":\"" << name << "\",";
    outputStream << "\"ph\":\"X\",";
    outputStream << "\"pid\":0,";
    outputStream << "\"tid\":" << seg.threadID << ",";
    outputStream << "\"ts\":" << seg.start;
    outputStream << "}";
    outputStream.flush();
  }

  void ProfileSession::begin_session(const std::string& name, const std::string& _filepath) {
    std::string filepath = se::Configuration::string_property("project_path") + _filepath;
    outputStream.open(filepath);
    write_header();
    currentSession = new InstrumentationSession(name);
  }

  void ProfileSession::end_session() {
    write_footer();
    outputStream.close();
    delete currentSession;
    currentSession = nullptr;
    profileCount = 0;
  }

  InstrumentationTimer::InstrumentationTimer(const char* name)
    : m_Name(name), m_Stopped(false) {
    m_StartTimepoint = std::chrono::high_resolution_clock::now();
  }

  InstrumentationTimer::~InstrumentationTimer() {
    if (!m_Stopped) stop();
  }

  void InstrumentationTimer::stop() {
    auto endTimepoint = std::chrono::high_resolution_clock::now();
    uint64_t start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint)
      .time_since_epoch().count();
    uint64_t end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint)
      .time_since_epoch().count();
    uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
    Singleton<ProfileSession>::instance()->write_segment({ std::string(m_Name), threadID, start, end });
    m_Stopped = true;
  }
}