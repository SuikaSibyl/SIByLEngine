#include "ex.tinyprbrtloader.hpp"
#include <map>
#include <set>
#include <thread>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <optional>
#include <functional>
#include <iostream>
#include <string_view>
#include <libdeflate/libdeflate.h>

namespace tiny_pbrt_loader {
  // Hack the pbrt code features
  using Float = double;
#define pstd std
#define CHECK(...)
#define LOG_FATAL(...)
#define CHECK_RARE(...)
#define LOG_VERBOSE(...) 
#define ErrorExit(...)
#define Warning(...)
#define Error(...)
#define CHECK_LT(...)
#define PBRT_CPU_GPU
#define CHECK_GE(...)
#define DCHECK_NE(...)
#define CHECK_EQ(...)
#define DCHECK_LT(...)
#define CHECK_GT(...)

  static std::string path_of_the_main_file = "";

  std::string ResolveFilename(std::string const& name) {
    return path_of_the_main_file + name;
  }

  // helpers, fwiw
  template <typename T>
  static auto operator<<(std::ostream& os, const T& v) -> decltype(v.ToString(), os) {
    return os << v.ToString();
  }
  template <typename T>
  static auto operator<<(std::ostream& os, const T& v) -> decltype(ToString(v), os) {
    return os << ToString(v);
  }

  template <typename T>
  inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<T>& p) {
    if (p)
      return os << p->ToString();
    else
      return os << "(nullptr)";
  }

  template <typename T>
  inline std::ostream& operator<<(std::ostream& os, const std::unique_ptr<T>& p) {
    if (p)
      return os << p->ToString();
    else
      return os << "(nullptr)";
  }

  namespace detail {
    std::string FloatToString(float v) {
      return std::to_string(v);
    }
    std::string DoubleToString(double v) {
      return std::to_string(v);
    }

    template <typename T>
    struct IntegerFormatTrait;

    template <>
    struct IntegerFormatTrait<bool> {
      static constexpr const char* fmt() { return "d"; }
    };
    template <>
    struct IntegerFormatTrait<char> {
      static constexpr const char* fmt() { return "d"; }
    };
    template <>
    struct IntegerFormatTrait<unsigned char> {
      static constexpr const char* fmt() { return "d"; }
    };
    template <>
    struct IntegerFormatTrait<int> {
      static constexpr const char* fmt() { return "d"; }
    };
    template <>
    struct IntegerFormatTrait<unsigned int> {
      static constexpr const char* fmt() { return "u"; }
    };
    template <>
    struct IntegerFormatTrait<short> {
      static constexpr const char* fmt() { return "d"; }
    };
    template <>
    struct IntegerFormatTrait<unsigned short> {
      static constexpr const char* fmt() { return "u"; }
    };
    template <>
    struct IntegerFormatTrait<long> {
      static constexpr const char* fmt() { return "ld"; }
    };
    template <>
    struct IntegerFormatTrait<unsigned long> {
      static constexpr const char* fmt() { return "lu"; }
    };
    template <>
    struct IntegerFormatTrait<long long> {
      static constexpr const char* fmt() { return "lld"; }
    };
    template <>
    struct IntegerFormatTrait<unsigned long long> {
      static constexpr const char* fmt() { return "llu"; }
    };

    template <typename T>
    using HasSize =
      std::is_integral<typename std::decay_t<decltype(std::declval<T&>().size())>>;

    template <typename T>
    using HasData =
      std::is_pointer<typename std::decay_t<decltype(std::declval<T&>().data())>>;

    // Don't use size()/data()-based operator<< for std::string...
    inline std::ostream& operator<<(std::ostream& os, const std::string& str) {
      return std::operator<<(os, str);
    }

    template <typename T>
    inline std::enable_if_t<HasSize<T>::value&& HasData<T>::value, std::ostream&>
      operator<<(std::ostream& os, const T& v) {
      os << "[ ";
      auto ptr = v.data();
      for (size_t i = 0; i < v.size(); ++i) {
        os << ptr[i];
        if (i < v.size() - 1)
          os << ", ";
      }
      return os << " ]";
    }

    // base case
    void stringPrintfRecursive(std::string* s, const char* fmt) {
      const char* c = fmt;
      // No args left; make sure there aren't any extra formatting
      // specifiers.
      while (*c) {
        if (*c == '%') {
          if (c[1] != '%')
            LOG_FATAL("Not enough optional values passed to Printf.");
          ++c;
        }
        *s += *c++;
      }
    }

    // 1. Copy from fmt to *s, up to the next formatting directive.
    // 2. Advance fmt past the next formatting directive and return the
    //    formatting directive as a string.
    std::string copyToFormatString(const char** fmt_ptr, std::string* s) {
      const char*& fmt = *fmt_ptr;
      while (*fmt) {
        if (*fmt != '%') {
          *s += *fmt;
          ++fmt;
        }
        else if (fmt[1] == '%') {
          // "%%"; let it pass through
          *s += '%';
          *s += '%';
          fmt += 2;
        }
        else
          // fmt is at the start of a formatting directive.
          break;
      }

      std::string nextFmt;
      while (*fmt) {
        char c = *fmt;
        nextFmt += c;
        ++fmt;
        // Is it a conversion specifier?
        if (c == 'd' || c == 'i' || c == 'o' || c == 'u' || c == 'x' || c == 'e' ||
          c == 'E' || c == 'f' || c == 'F' || c == 'g' || c == 'G' || c == 'a' ||
          c == 'A' || c == 'c' || c == 'C' || c == 's' || c == 'S' || c == 'p')
          break;
      }

      return nextFmt;
    }

    template <typename T>
    inline typename std::enable_if_t<!std::is_class_v<typename std::decay_t<T>>, std::string>
      formatOne(const char* fmt, T&& v) {
      // Figure out how much space we need to allocate; add an extra
      // character for the '\0'.
      size_t size = snprintf(nullptr, 0, fmt, v) + 1;
      std::string str;
      str.resize(size);
      snprintf(&str[0], size, fmt, v);
      str.pop_back();  // remove trailing NUL
      return str;
    }

    template <typename T>
    inline typename std::enable_if_t<std::is_class_v<typename std::decay_t<T>>, std::string>
      formatOne(const char* fmt, T&& v) {
      LOG_FATAL("Printf: Non-basic type %s passed for format string %s", typeid(v).name(),
        fmt);
      return "";
    }

    template <typename T, typename... Args>
    inline void stringPrintfRecursive(std::string* s, const char* fmt, T&& v, Args &&...args);

    template <typename T, typename... Args>
    inline void stringPrintfRecursiveWithPrecision(std::string* s, const char* fmt,
      const std::string& nextFmt, T&& v,
      Args &&...args) {
      LOG_FATAL("MEH");
    }

    template <typename T, typename... Args>
    inline typename std::enable_if_t<!std::is_class_v<typename std::decay_t<T>>, void>
      stringPrintfRecursiveWithPrecision(std::string* s, const char* fmt,
        const std::string& nextFmt, int precision, T&& v,
        Args &&...args) {
      size_t size = snprintf(nullptr, 0, nextFmt.c_str(), precision, v) + 1;
      std::string str;
      str.resize(size);
      snprintf(&str[0], size, nextFmt.c_str(), precision, v);
      str.pop_back();  // remove trailing NUL
      *s += str;

      stringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
    }

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4102)  // bogus "unreferenced label" warning for done: below
#endif

    // General-purpose version of stringPrintfRecursive; add the formatted
    // output for a single StringPrintf() argument to the final result string
    // in *s.
    template <typename T, typename... Args>
    inline void stringPrintfRecursive(std::string* s, const char* fmt, T&& v,
      Args &&...args) {
      std::string nextFmt = copyToFormatString(&fmt, s);
      bool precisionViaArg = nextFmt.find('*') != std::string::npos;

      bool isSFmt = nextFmt.find('s') != std::string::npos;
      bool isDFmt = nextFmt.find('d') != std::string::npos;

      if constexpr (std::is_integral_v<std::decay_t<T>>) {
        if (precisionViaArg) {
          stringPrintfRecursiveWithPrecision(s, fmt, nextFmt, v,
            std::forward<Args>(args)...);
          return;
        }
      }
      else if (precisionViaArg)
        LOG_FATAL("Non-integral type provided for %* format.");

      if constexpr (std::is_same_v<std::decay_t<T>, float>)
        if (nextFmt == "%f" || nextFmt == "%s") {
          *s += detail::FloatToString(v);
          goto done;
        }

      if constexpr (std::is_same_v<std::decay_t<T>, double>)
        if (nextFmt == "%f" || nextFmt == "%s") {
          *s += detail::DoubleToString(v);
          goto done;
        }

      if constexpr (std::is_same_v<std::decay_t<T>, bool>)  // FIXME: %-10s with bool
        if (isSFmt) {
          *s += bool(v) ? "true" : "false";
          goto done;
        }

      if constexpr (std::is_integral_v<std::decay_t<T>>) {
        if (isDFmt) {
          nextFmt.replace(nextFmt.find('d'), 1,
            detail::IntegerFormatTrait<std::decay_t<T>>::fmt());
          *s += formatOne(nextFmt.c_str(), std::forward<T>(v));
          goto done;
        }
      }
      else if (isDFmt)
        LOG_FATAL("Non-integral type passed to %d format.");

      if (isSFmt) {
        std::stringstream ss;
        ss << v;
        *s += formatOne(nextFmt.c_str(), ss.str().c_str());
      }
      else if (!nextFmt.empty())
        *s += formatOne(nextFmt.c_str(), std::forward<T>(v));
      else
        LOG_FATAL("Excess values passed to Printf.");

    done:
      stringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
    }

#ifdef _MSC_VER
#pragma warning(pop)
#endif
  }

  // Printing Function Declarations
  template <typename... Args>
  void Printf(const char* fmt, Args &&...args);
  template <typename... Args>
  inline std::string StringPrintf(const char* fmt, Args &&...args);

  template <typename... Args>
  inline std::string StringPrintf(const char* fmt, Args &&...args) {
    std::string ret;
    detail::stringPrintfRecursive(&ret, fmt, std::forward<Args>(args)...);
    return ret;
  }

  template <typename... Args>
  void Printf(const char* fmt, Args &&...args) {
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    fputs(s.c_str(), stdout);
  }

  std::string FileLoc::ToString() const {
    return StringPrintf("%s:%d:%d", std::string(filename.data(), filename.size()), line,
      column);
  }

  template <typename T, int N, class Allocator = std::pmr::polymorphic_allocator<T>>
  class InlinedVector {
  public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    InlinedVector(const Allocator& alloc = {}) : alloc(alloc) {}
    InlinedVector(size_t count, const T& value, const Allocator& alloc = {})
      : alloc(alloc) {
      reserve(count);
      for (size_t i = 0; i < count; ++i)
        this->alloc.template construct<T>(begin() + i, value);
      nStored = count;
    }
    InlinedVector(size_t count, const Allocator& alloc = {})
      : InlinedVector(count, T{}, alloc) {}
    InlinedVector(const InlinedVector& other, const Allocator& alloc = {})
      : alloc(alloc) {
      reserve(other.size());
      for (size_t i = 0; i < other.size(); ++i)
        this->alloc.template construct<T>(begin() + i, other[i]);
      nStored = other.size();
    }
    template <class InputIt>
    InlinedVector(InputIt first, InputIt last, const Allocator& alloc = {})
      : alloc(alloc) {
      reserve(last - first);
      for (InputIt iter = first; iter != last; ++iter, ++nStored)
        this->alloc.template construct<T>(begin() + nStored, *iter);
    }
    InlinedVector(InlinedVector&& other) : alloc(other.alloc) {
      nStored = other.nStored;
      nAlloc = other.nAlloc;
      ptr = other.ptr;
      if (other.nStored <= N)
        for (int i = 0; i < other.nStored; ++i)
          alloc.template construct<T>(fixed + i, std::move(other.fixed[i]));
      // Leave other.nStored as is, so that the detrius left after we
      // moved out of fixed has its destructors run...
      else
        other.nStored = 0;

      other.nAlloc = 0;
      other.ptr = nullptr;
    }
    InlinedVector(InlinedVector&& other, const Allocator& alloc) {
      LOG_FATAL("TODO");

      if (alloc == other.alloc) {
        ptr = other.ptr;
        nAlloc = other.nAlloc;
        nStored = other.nStored;
        if (other.nStored <= N)
          for (int i = 0; i < other.nStored; ++i)
            fixed[i] = std::move(other.fixed[i]);

        other.ptr = nullptr;
        other.nAlloc = other.nStored = 0;
      }
      else {
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
          alloc.template construct<T>(begin() + i, std::move(other[i]));
        nStored = other.size();
      }
    }
    InlinedVector(std::initializer_list<T> init, const Allocator& alloc = {})
      : InlinedVector(init.begin(), init.end(), alloc) {}

    InlinedVector& operator=(const InlinedVector& other) {
      if (this == &other)
        return *this;

      clear();
      reserve(other.size());
      for (size_t i = 0; i < other.size(); ++i)
        alloc.template construct<T>(begin() + i, other[i]);
      nStored = other.size();

      return *this;
    }
    InlinedVector& operator=(InlinedVector&& other) {
      if (this == &other)
        return *this;

      clear();
      if (alloc == other.alloc) {
        pstd::swap(ptr, other.ptr);
        pstd::swap(nAlloc, other.nAlloc);
        pstd::swap(nStored, other.nStored);
        if (nStored > 0 && !ptr) {
          for (int i = 0; i < nStored; ++i)
            alloc.template construct<T>(fixed + i, std::move(other.fixed[i]));
          other.nStored = nStored;  // so that dtors run...
        }
      }
      else {
        reserve(other.size());
        for (size_t i = 0; i < other.size(); ++i)
          alloc.template construct<T>(begin() + i, std::move(other[i]));
        nStored = other.size();
      }

      return *this;
    }
    InlinedVector& operator=(std::initializer_list<T>& init) {
      clear();
      reserve(init.size());
      for (const auto& value : init) {
        alloc.template construct<T>(begin() + nStored, value);
        ++nStored;
      }
      return *this;
    }

    void assign(size_type count, const T& value) {
      clear();
      reserve(count);
      for (size_t i = 0; i < count; ++i)
        alloc.template construct<T>(begin() + i, value);
      nStored = count;
    }
    template <class InputIt>
    void assign(InputIt first, InputIt last) {
      // TODO
      LOG_FATAL("TODO");
    }
    void assign(std::initializer_list<T>& init) { assign(init.begin(), init.end()); }

    ~InlinedVector() {
      clear();
      alloc.deallocate_object(ptr, nAlloc);
    }

    iterator begin() { return ptr ? ptr : fixed; }
    iterator end() { return begin() + nStored; }
    const_iterator begin() const { return ptr ? ptr : fixed; }
    const_iterator end() const { return begin() + nStored; }
    const_iterator cbegin() const { return ptr ? ptr : fixed; }
    const_iterator cend() const { return begin() + nStored; }

    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    allocator_type get_allocator() const { return alloc; }
    size_t size() const { return nStored; }
    bool empty() const { return size() == 0; }
    size_t max_size() const { return (size_t)-1; }
    size_t capacity() const { return ptr ? nAlloc : N; }

    void reserve(size_t n) {
      if (capacity() >= n)
        return;

      T* ra = alloc.template allocate_object<T>(n);
      for (int i = 0; i < nStored; ++i) {
        alloc.template construct<T>(ra + i, std::move(begin()[i]));
        alloc.destroy(begin() + i);
      }

      alloc.deallocate_object(ptr, nAlloc);
      nAlloc = n;
      ptr = ra;
    }
    // TODO: shrink_to_fit

    reference operator[](size_type index) {
      DCHECK_LT(index, size());
      return begin()[index];
    }
    const_reference operator[](size_type index) const {
      DCHECK_LT(index, size());
      return begin()[index];
    }
    reference front() { return *begin(); }
    const_reference front() const { return *begin(); }

    reference back() { return *(begin() + nStored - 1); }
    const_reference back() const { return *(begin() + nStored - 1); }
    pointer data() { return ptr ? ptr : fixed; }
    const_pointer data() const { return ptr ? ptr : fixed; }

    void clear() {
      for (int i = 0; i < nStored; ++i)
        alloc.destroy(begin() + i);
      nStored = 0;
    }

    iterator insert(const_iterator, const T& value) {
      // TODO
      LOG_FATAL("TODO");
    }
    iterator insert(const_iterator, T&& value) {
      // TODO
      LOG_FATAL("TODO");
    }
    iterator insert(const_iterator pos, size_type count, const T& value) {
      // TODO
      LOG_FATAL("TODO");
    }
    template <class InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last) {
      if (pos == end()) {
        reserve(size() + (last - first));
        iterator pos = end();
        for (auto iter = first; iter != last; ++iter, ++pos)
          alloc.template construct<T>(pos, *iter);
        nStored += last - first;
        return pos;
      }
      else {
        // TODO
        LOG_FATAL("TODO");
      }
    }
    iterator insert(const_iterator pos, std::initializer_list<T> init) {
      // TODO
      LOG_FATAL("TODO");
    }

    template <class... Args>
    iterator emplace(const_iterator pos, Args &&...args) {
      // TODO
      LOG_FATAL("TODO");
    }
    template <class... Args>
    void emplace_back(Args &&...args) {
      // TODO
      LOG_FATAL("TODO");
    }

    iterator erase(const_iterator cpos) {
      iterator pos =
        begin() + (cpos - begin());  // non-const iterator, thank you very much
      while (pos != end() - 1) {
        *pos = std::move(*(pos + 1));
        ++pos;
      }
      alloc.destroy(pos);
      --nStored;
      return begin() + (cpos - begin());
    }
    iterator erase(const_iterator first, const_iterator last) {
      // TODO
      LOG_FATAL("TODO");
    }

    void push_back(const T& value) {
      if (size() == capacity())
        reserve(2 * capacity());

      alloc.construct(begin() + nStored, value);
      ++nStored;
    }
    void push_back(T&& value) {
      if (size() == capacity())
        reserve(2 * capacity());

      alloc.construct(begin() + nStored, std::move(value));
      ++nStored;
    }
    void pop_back() {
      DCHECK(!empty());
      alloc.destroy(begin() + nStored - 1);
      --nStored;
    }

    void resize(size_type n) {
      if (n < size()) {
        for (size_t i = n; n < size(); ++i)
          alloc.destroy(begin() + i);
      }
      else if (n > size()) {
        reserve(n);
        for (size_t i = nStored; i < n; ++i)
          alloc.construct(begin() + i);
      }
      nStored = n;
    }
    void resize(size_type count, const value_type& value) {
      // TODO
      LOG_FATAL("TODO");
    }

    void swap(InlinedVector& other) {
      // TODO
      LOG_FATAL("TODO");
    }

    std::vector<T> finalize() {
      std::vector<T> vec(size());
      memcpy(vec.data(), data(), size() * sizeof(T));
      return vec;
    }

  private:
    Allocator alloc;
    // ptr non-null is discriminator for whether fixed[] is valid...
    T* ptr = nullptr;
    union {
      T fixed[N];
    };
    size_t nAlloc = 0, nStored = 0;
  };

  using ParsedParameterVector = InlinedVector<ParsedParameter*, 8>;

  class ParserTarget {
  public:
    virtual void Scale(Float sx, Float sy, Float sz, FileLoc loc) = 0;
    virtual void Shape(const std::string& name,
      ParsedParameterVector params, FileLoc loc) = 0;
    virtual ~ParserTarget();

    virtual void Option(const std::string& name, const std::string& value,
      FileLoc loc) = 0;

    virtual void Identity(FileLoc loc) = 0;
    virtual void Translate(Float dx, Float dy, Float dz, FileLoc loc) = 0;
    virtual void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc) = 0;
    virtual void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
      Float ux, Float uy, Float uz, FileLoc loc) = 0;
    virtual void ConcatTransform(Float transform[16], FileLoc loc) = 0;
    virtual void Transform(Float transform[16], FileLoc loc) = 0;
    virtual void CoordinateSystem(const std::string&, FileLoc loc) = 0;
    virtual void CoordSysTransform(const std::string&, FileLoc loc) = 0;
    virtual void ActiveTransformAll(FileLoc loc) = 0;
    virtual void ActiveTransformEndTime(FileLoc loc) = 0;
    virtual void ActiveTransformStartTime(FileLoc loc) = 0;
    virtual void TransformTimes(Float start, Float end, FileLoc loc) = 0;

    virtual void ColorSpace(const std::string& n, FileLoc loc) = 0;
    virtual void PixelFilter(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void Film(const std::string& type, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void Accelerator(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void Integrator(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void Camera(const std::string&, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void MakeNamedMedium(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void MediumInterface(const std::string& insideName,
      const std::string& outsideName, FileLoc loc) = 0;
    virtual void Sampler(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;

    virtual void WorldBegin(FileLoc loc) = 0;
    virtual void AttributeBegin(FileLoc loc) = 0;
    virtual void AttributeEnd(FileLoc loc) = 0;
    virtual void Attribute(const std::string& target, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void Texture(const std::string& name, const std::string& type,
      const std::string& texname, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void Material(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void MakeNamedMaterial(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void NamedMaterial(const std::string& name, FileLoc loc) = 0;
    virtual void LightSource(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void AreaLightSource(const std::string& name, ParsedParameterVector params,
      FileLoc loc) = 0;
    virtual void ReverseOrientation(FileLoc loc) = 0;
    virtual void ObjectBegin(const std::string& name, FileLoc loc) = 0;
    virtual void ObjectEnd(FileLoc loc) = 0;
    virtual void ObjectInstance(const std::string& name, FileLoc loc) = 0;

    virtual void EndOfFiles() = 0;

  protected:
    template <typename... Args>
    void ErrorExitDeferred(const char* fmt, Args &&... args) const {
      errorExit = true;
      Error(fmt, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void ErrorExitDeferred(const FileLoc* loc, const char* fmt, Args &&... args) const {
      errorExit = true;
      Error(loc, fmt, std::forward<Args>(args)...);
    }

    mutable bool errorExit = false;
  };

  void ParsedParameter::AddFloat(Float v) {
    CHECK(ints.empty() && strings.empty() && bools.empty());
    floats.push_back(v);
  }

  void ParsedParameter::AddInt(int i) {
    CHECK(floats.empty() && strings.empty() && bools.empty());
    ints.push_back(i);
  }

  void ParsedParameter::AddString(std::string_view str) {
    CHECK(floats.empty() && ints.empty() && bools.empty());
    strings.push_back({ str.begin(), str.end() });
  }

  void ParsedParameter::AddBool(bool v) {
    CHECK(floats.empty() && ints.empty() && strings.empty());
    bools.push_back(v);
  }

  std::string ParsedParameter::ToString() const {
    std::string str;
    str += std::string("\"") + type + " " + name + std::string("\" [ ");
    if (!floats.empty())
      for (Float d : floats)
        str += StringPrintf("%f ", d);
    else if (!ints.empty())
      for (int i : ints)
        str += StringPrintf("%d ", i);
    else if (!strings.empty())
      for (const auto& s : strings)
        str += '\"' + s + "\" ";
    else if (!bools.empty())
      for (bool b : bools)
        str += b ? "true " : "false ";
    str += "] ";

    return str;
  }

  ParserTarget::~ParserTarget() {}


  // Token Definition
  struct Token {
    Token() = default;
    Token(std::string_view token, FileLoc loc) : token(token), loc(loc) {}
    std::string ToString() const;
    std::string_view token;
    FileLoc loc;
  };

  // Tokenizer Definition
  class Tokenizer {
  public:
    // Tokenizer Public Methods
    Tokenizer(std::string str, std::string filename,
      std::function<void(const char*, const FileLoc*)> errorCallback);
#if defined(PBRT_HAVE_MMAP) || defined(PBRT_IS_WINDOWS)
    Tokenizer(void* ptr, size_t len, std::string filename,
      std::function<void(const char*, const FileLoc*)> errorCallback);
#endif
    ~Tokenizer();

    static std::unique_ptr<Tokenizer> CreateFromFile(
      const std::string& filename,
      std::function<void(const char*, const FileLoc*)> errorCallback);
    static std::unique_ptr<Tokenizer> CreateFromString(
      std::string str,
      std::function<void(const char*, const FileLoc*)> errorCallback);

    std::optional<Token> Next();

    // Just for parse().
    // TODO? Have a method to set this?
    FileLoc loc;

  private:
    // Tokenizer Private Methods
    void CheckUTF(const void* ptr, int len) const;

    int getChar() {
      if (pos == end)
        return EOF;
      int ch = *pos++;
      if (ch == '\n') {
        ++loc.line;
        loc.column = 0;
      }
      else
        ++loc.column;
      return ch;
    }
    void ungetChar() {
      --pos;
      if (*pos == '\n')
        // Don't worry about the column; we'll be going to the start of
        // the next line again shortly...
        --loc.line;
    }

    // Tokenizer Private Members
    // This function is called if there is an error during lexing.
    std::function<void(const char*, const FileLoc*)> errorCallback;

#if defined(PBRT_HAVE_MMAP) || defined(PBRT_IS_WINDOWS)
    // Scene files on disk are mapped into memory for lexing.  We need to
    // hold on to the starting pointer and total length so they can be
    // unmapped in the destructor.
    void* unmapPtr = nullptr;
    size_t unmapLength = 0;
#endif

    // If the input is stdin, then we copy everything until EOF into this
    // string and then start lexing.  This is a little wasteful (versus
    // tokenizing directly from stdin), but makes the implementation
    // simpler.
    std::string contents;

    // Pointers to the current position in the file and one past the end of
    // the file.
    const char* pos, * end;

    // If there are escaped characters in the string, we can't just return
    // a std::string_view into the mapped file. In that case, we handle the
    // escaped characters and return a std::string_view to sEscaped.  (And
    // thence, std::string_views from previous calls to Next() must be invalid
    // after a subsequent call, since we may reuse sEscaped.)
    std::string sEscaped;
  };

  std::string ReadFileContents(std::string filename) {
#ifdef _WIN32
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if (!ifs)
      ErrorExit("%s: %s", filename, ErrorString());
    return std::string((std::istreambuf_iterator<char>(ifs)),
      (std::istreambuf_iterator<char>()));
#else
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
      ErrorExit("%s: %s", filename, ErrorString());

    struct stat stat;
    if (fstat(fd, &stat) != 0)
      ErrorExit("%s: %s", filename, ErrorString());

    std::string contents(stat.st_size, '\0');
    if (read(fd, contents.data(), stat.st_size) == -1)
      ErrorExit("%s: %s", filename, ErrorString());

    close(fd);
    return contents;
#endif
  }

  std::string ReadDecompressedFileContents(std::string filename) {
    std::string compressed = ReadFileContents(filename);

    // Get the size of the uncompressed file: with gzip, it's stored in the
    // last 4 bytes of the file.  (One nit is that only 4 bytes are used,
    // so it's actually the uncompressed size mod 2^32.)
    CHECK_GT(compressed.size(), 4);
    size_t sizeOffset = compressed.size() - 4;

    // It's stored in little-endian, so manually reconstruct the value to
    // be sure that it ends up in the right order for the target system.
    const unsigned char* s = (const unsigned char*)compressed.data() + sizeOffset;
    size_t size = (uint32_t(s[0]) | (uint32_t(s[1]) << 8) | (uint32_t(s[2]) << 16) |
      (uint32_t(s[3]) << 24));

    libdeflate_decompressor* d = libdeflate_alloc_decompressor();
    std::string decompressed(size, '\0');
    int retries = 0;
    while (true) {
      size_t actualOut;
      libdeflate_result result = libdeflate_gzip_decompress(
        d, compressed.data(), compressed.size(), decompressed.data(),
        decompressed.size(), &actualOut);
      switch (result) {
      case LIBDEFLATE_SUCCESS:
        CHECK_EQ(actualOut, decompressed.size());
        LOG_VERBOSE("Decompressed %s from %d to %d bytes", filename,
          compressed.size(), decompressed.size());
        return decompressed;

      case LIBDEFLATE_BAD_DATA:
        ErrorExit("%s: invalid or corrupt compressed data", filename);

      case LIBDEFLATE_INSUFFICIENT_SPACE:
        // Assume that the decompressed contents are > 4GB and that
        // thus the size reported in the file didn't tell the whole
        // story.  Since the stored size is mod 2^32, try increasing
        // the allocation by that much.
        decompressed.resize(decompressed.size() + (1ull << 32));

        // But if we keep going around in circles, then fail eventually
        // since there is probably some other problem.
        CHECK_LT(++retries, 10);
        break;

      default:
      case LIBDEFLATE_SHORT_OUTPUT:
        // This should never be returned by libdeflate, since we are
        // passing a non-null actualOut pointer...
        LOG_FATAL("Unexpected return value from libdeflate");
      }
    }
  }

  std::unique_ptr<Tokenizer> Tokenizer::CreateFromFile(
    const std::string& filename,
    std::function<void(const char*, const FileLoc*)> errorCallback) {
    if (filename == "-") {
      // Handle stdin by slurping everything into a string.
      std::string str;
      int ch;
      while ((ch = getchar()) != EOF)
        str.push_back((char)ch);
      return std::make_unique<Tokenizer>(std::move(str), "<stdin>",
        std::move(errorCallback));
    }

    if (filename.substr(filename.size() - 3) == ".gz") {
      std::string str = ReadDecompressedFileContents(filename);
      return std::make_unique<Tokenizer>(std::move(str), filename,
        std::move(errorCallback));
    }
    else {
      std::string str = ReadFileContents(filename);
      return std::make_unique<Tokenizer>(std::move(str), filename,
        std::move(errorCallback));
    }
  }

  std::unique_ptr<Tokenizer> Tokenizer::CreateFromString(
    std::string str, std::function<void(const char*, const FileLoc*)> errorCallback) {
    return std::make_unique<Tokenizer>(std::move(str), "<stdin>",
      std::move(errorCallback));
  }

  Tokenizer::Tokenizer(std::string str, std::string filename,
    std::function<void(const char*, const FileLoc*)> errorCallback)
    : errorCallback(std::move(errorCallback)), contents(std::move(str)) {
    loc = FileLoc(*new std::string(filename));
    pos = contents.data();
    end = pos + contents.size();
    //tokenizerMemory += contents.size();
    CheckUTF(contents.data(), contents.size());
  }

  Tokenizer::~Tokenizer() {}

  void Tokenizer::CheckUTF(const void* ptr, int len) const {
    const unsigned char* c = (const unsigned char*)ptr;
    // https://en.wikipedia.org/wiki/Byte_order_mark
    if (len >= 2 && ((c[0] == 0xfe && c[1] == 0xff) || (c[0] == 0xff && c[1] == 0xfe)))
      errorCallback("File is encoded with UTF-16, which is not currently "
        "supported by pbrt (https://github.com/mmp/pbrt-v4/issues/136).",
        &loc);
  }

  // Tokenizer Implementation
  static char decodeEscaped(int ch, const FileLoc& loc) {
    switch (ch) {
    case EOF:
      ErrorExit(&loc, "premature EOF after character escape '\\'");
    case 'b':
      return '\b';
    case 'f':
      return '\f';
    case 'n':
      return '\n';
    case 'r':
      return '\r';
    case 't':
      return '\t';
    case '\\':
      return '\\';
    case '\'':
      return '\'';
    case '\"':
      return '\"';
    default:
      ErrorExit(&loc, "unexpected escaped character \"%c\"", ch);
    }
    return 0;  // NOTREACHED
  }

  pstd::optional<Token> Tokenizer::Next() {
    while (true) {
      const char* tokenStart = pos;
      FileLoc startLoc = loc;

      int ch = getChar();
      if (ch == EOF)
        return {};
      else if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') {
        // nothing
      }
      else if (ch == '"') {
        // scan to closing quote
        bool haveEscaped = false;
        while ((ch = getChar()) != '"') {
          if (ch == EOF) {
            errorCallback("premature EOF", &startLoc);
            return {};
          }
          else if (ch == '\n') {
            errorCallback("unterminated string", &startLoc);
            return {};
          }
          else if (ch == '\\') {
            haveEscaped = true;
            // Grab the next character
            if ((ch = getChar()) == EOF) {
              errorCallback("premature EOF", &startLoc);
              return {};
            }
          }
        }

        if (!haveEscaped)
          return Token({ tokenStart, size_t(pos - tokenStart) }, startLoc);
        else {
          sEscaped.clear();
          for (const char* p = tokenStart; p < pos; ++p) {
            if (*p != '\\')
              sEscaped.push_back(*p);
            else {
              ++p;
              CHECK_LT(p, pos);
              sEscaped.push_back(decodeEscaped(*p, startLoc));
            }
          }
          return Token({ sEscaped.data(), sEscaped.size() }, startLoc);
        }
      }
      else if (ch == '[' || ch == ']') {
        return Token({ tokenStart, size_t(1) }, startLoc);
      }
      else if (ch == '#') {
        // comment: scan to EOL (or EOF)
        while ((ch = getChar()) != EOF) {
          if (ch == '\n' || ch == '\r') {
            ungetChar();
            break;
          }
        }

        return Token({ tokenStart, size_t(pos - tokenStart) }, startLoc);
      }
      else {
        // Regular statement or numeric token; scan until we hit a
        // space, opening quote, or bracket.
        while ((ch = getChar()) != EOF) {
          if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r' || ch == '"' ||
            ch == '[' || ch == ']') {
            ungetChar();
            break;
          }
        }
        return Token({ tokenStart, size_t(pos - tokenStart) }, startLoc);
      }
    }
  }

  constexpr int TokenOptional = 0;
  constexpr int TokenRequired = 1;

  static std::string toString(std::string_view s) {
    return std::string(s.data(), s.size());
  }

  std::string Token::ToString() const {
    return StringPrintf("[ Token token: %s loc: %s ]", toString(token), loc);
  }

  static int parseInt(const Token& t) {
    bool negate = t.token[0] == '-';

    int index = 0;
    if (t.token[0] == '+' || t.token[0] == '-')
      ++index;

    int64_t value = 0;
    while (index < t.token.size()) {
      if (!(t.token[index] >= '0' && t.token[index] <= '9'))
        ErrorExit(&t.loc, "\"%c\": expected a number", t.token[index]);
      value = 10 * value + (t.token[index] - '0');
      ++index;

      if (value > std::numeric_limits<int>::max())
        ErrorExit(&t.loc,
          "Numeric value too large to represent as a 32-bit integer.");
      else if (value < std::numeric_limits<int>::lowest())
        Warning(&t.loc, "Numeric value %d too low to represent as a 32-bit integer.");
    }

    return negate ? -value : value;
  }

  static double parseFloat(const Token& t) {
    // Fast path for a single digit
    if (t.token.size() == 1) {
      if (!(t.token[0] >= '0' && t.token[0] <= '9'))
        ErrorExit(&t.loc, "\"%c\": expected a number", t.token[0]);
      return t.token[0] - '0';
    }

    // Copy to a buffer so we can NUL-terminate it, as strto[idf]() expect.
    char buf[64];
    char* bufp = buf;
    std::unique_ptr<char[]> allocBuf;
    CHECK_RARE(1e-5, t.token.size() + 1 >= sizeof(buf));
    if (t.token.size() + 1 >= sizeof(buf)) {
      // This should be very unusual, but is necessary in case we get a
      // goofball number with lots of leading zeros, for example.
      allocBuf = std::make_unique<char[]>(t.token.size() + 1);
      bufp = allocBuf.get();
    }

    std::copy(t.token.begin(), t.token.end(), bufp);
    bufp[t.token.size()] = '\0';

    // Can we just use strtol?
    auto isInteger = [](std::string_view str) {
      for (char ch : str)
        if (!(ch >= '0' && ch <= '9'))
          return false;
      return true;
    };

    int length = 0;
    double val = 0;
    if (isInteger(t.token)) {
      char* endptr;
      val = double(strtol(bufp, &endptr, 10));
      length = endptr - bufp;
    }
    try {
      val = std::stod(std::string(bufp, t.token.size()));
    }
    catch (const std::invalid_argument& e) {
      std::cerr << "Invalid argument: " << e.what() << std::endl;
    }
    catch (const std::out_of_range& e) {
      std::cerr << "Out of range: " << e.what() << std::endl;
    }
    return val;
  }

  inline bool isQuotedString(std::string_view str) {
    return str.size() >= 2 && str[0] == '"' && str.back() == '"';
  }

  static std::string_view dequoteString(const Token& t) {
    if (!isQuotedString(t.token))
      ErrorExit(&t.loc, "\"%s\": expected quoted string", toString(t.token));

    std::string_view str = t.token;
    str.remove_prefix(1);
    str.remove_suffix(1);
    return str;
  }

  template <typename Next, typename Unget>
  static ParsedParameterVector parseParameters(
    Next nextToken, Unget ungetToken, bool formatting,
    const std::function<void(const Token& token, const char*)>& errorCallback) {
    ParsedParameterVector parameterVector;

    while (true) {
      pstd::optional<Token> t = nextToken(TokenOptional);
      if (!t.has_value())
        return parameterVector;

      if (!isQuotedString(t->token)) {
        ungetToken(*t);
        return parameterVector;
      }

      ParsedParameter* param = new ParsedParameter(t->loc);

      std::string_view decl = dequoteString(*t);

      auto skipSpace = [&decl](std::string_view::const_iterator iter) {
        while (iter != decl.end() && (*iter == ' ' || *iter == '\t'))
          ++iter;
        return iter;
      };
      // Skip to the next whitespace character (or the end of the string).
      auto skipToSpace = [&decl](std::string_view::const_iterator iter) {
        while (iter != decl.end() && *iter != ' ' && *iter != '\t')
          ++iter;
        return iter;
      };

      auto typeBegin = skipSpace(decl.begin());
      if (typeBegin == decl.end())
        ErrorExit(&t->loc, "Parameter \"%s\" doesn't have a type declaration?!",
          std::string(decl.begin(), decl.end()));

      // Find end of type declaration
      auto typeEnd = skipToSpace(typeBegin);
      param->type.assign(typeBegin, typeEnd);

      if (formatting) {  // close enough: upgrade...
        if (param->type == "point")
          param->type = "point3";
        if (param->type == "vector")
          param->type = "vector3";
        if (param->type == "color")
          param->type = "rgb";
      }

      auto nameBegin = skipSpace(typeEnd);
      if (nameBegin == decl.end())
        ErrorExit(&t->loc, "Unable to find parameter name from \"%s\"",
          std::string(decl.begin(), decl.end()));

      auto nameEnd = skipToSpace(nameBegin);
      param->name.assign(nameBegin, nameEnd);

      enum ValType { Unknown, String, Bool, Float, Int } valType = Unknown;

      if (param->type == "integer")
        valType = Int;

      auto addVal = [&](const Token& t) {
        if (isQuotedString(t.token)) {
          switch (valType) {
          case Unknown:
            valType = String;
            break;
          case String:
            break;
          case Float:
            errorCallback(t, "expected floating-point value");
          case Int:
            errorCallback(t, "expected integer value");
          case Bool:
            errorCallback(t, "expected Boolean value");
          }

          param->AddString(dequoteString(t));
        }
        else if (t.token[0] == 't' && t.token == "true") {
          switch (valType) {
          case Unknown:
            valType = Bool;
            break;
          case String:
            errorCallback(t, "expected string value");
          case Float:
            errorCallback(t, "expected floating-point value");
          case Int:
            errorCallback(t, "expected integer value");
          case Bool:
            break;
          }

          param->AddBool(true);
        }
        else if (t.token[0] == 'f' && t.token == "false") {
          switch (valType) {
          case Unknown:
            valType = Bool;
            break;
          case String:
            errorCallback(t, "expected string value");
          case Float:
            errorCallback(t, "expected floating-point value");
          case Int:
            errorCallback(t, "expected integer value");
          case Bool:
            break;
          }

          param->AddBool(false);
        }
        else {
          switch (valType) {
          case Unknown:
            valType = Float;
            break;
          case String:
            errorCallback(t, "expected string value");
          case Float:
            break;
          case Int:
            break;
          case Bool:
            errorCallback(t, "expected Boolean value");
          }

          if (valType == Int)
            param->AddInt(parseInt(t));
          else
            param->AddFloat(parseFloat(t));
        }
      };

      Token val = *nextToken(TokenRequired);

      if (val.token == "[") {
        while (true) {
          val = *nextToken(TokenRequired);
          if (val.token == "]")
            break;
          addVal(val);
        }
      }
      else {
        addVal(val);
      }

      if (formatting && param->type == "bool") {
        for (const auto& b : param->strings) {
          if (b == "true")
            param->bools.push_back(true);
          else if (b == "false")
            param->bools.push_back(false);
          else
            Error(&param->loc,
              "%s: neither \"true\" nor \"false\" in bool "
              "parameter list.",
              b);
        }
        param->strings.clear();
      }

      parameterVector.push_back(param);
    }

    return parameterVector;
  }

  void parse(ParserTarget* target, std::unique_ptr<Tokenizer> t) {
    static std::atomic<bool> warnedTransformBeginEndDeprecated{ false };

    //std::vector<std::pair<AsyncJob<int>*, BasicSceneBuilder*>> imports;

    std::vector<std::unique_ptr<Tokenizer>> fileStack;
    fileStack.push_back(std::move(t));

    std::optional<Token> ungetToken;

    auto parseError = [&](const char* msg, const FileLoc* loc) {
      ErrorExit(loc, "%s", msg);
    };

    // nextToken is a little helper function that handles the file stack,
    // returning the next token from the current file until reaching EOF,
    // at which point it switches to the next file (if any).
    std::function<std::optional<Token>(int)> nextToken;
    nextToken = [&](int flags) -> std::optional<Token> {
      if (ungetToken.has_value())
        return std::exchange(ungetToken, {});

      if (fileStack.empty()) {
        if ((flags & TokenRequired) != 0) {
          ErrorExit("premature end of file");
        }
        return {};
      }

      std::optional<Token> tok = fileStack.back()->Next();

      if (!tok) {
        // We've reached EOF in the current file. Anything more to parse?
        LOG_VERBOSE("Finished parsing %s",
          std::string(fileStack.back()->loc.filename.begin(),
            fileStack.back()->loc.filename.end()));
        fileStack.pop_back();
        return nextToken(flags);
      }
      else if (tok->token[0] == '#') {
        // Swallow comments, unless --format or --toply was given, in
        // which case they're printed to stdout.
        return nextToken(flags);
      }
      else
        // Regular token; success.
        return tok;
    };

    auto unget = [&](Token t) {
      CHECK(!ungetToken.has_value());
      ungetToken = t;
    };

    // Helper function for pbrt API entrypoints that take a single string
    // parameter and a ParameterVector (e.g. pbrtShape()).
    auto basicParamListEntrypoint =
      [&](void (ParserTarget::* apiFunc)(const std::string&, ParsedParameterVector,
        FileLoc),
        FileLoc loc) {
          Token t = *nextToken(TokenRequired);
          std::string_view dequoted = dequoteString(t);
          std::string n = toString(dequoted);
          ParsedParameterVector parameterVector = parseParameters(
            nextToken, unget, false, [&](const Token& t, const char* msg) {
              std::string token = toString(t.token);
          std::string str = StringPrintf("%s: %s", token, msg);
          parseError(str.c_str(), &t.loc);
            });
          (target->*apiFunc)(n, std::move(parameterVector), loc);
    };

    auto syntaxError = [&](const Token& t) {
      ErrorExit(&t.loc, "Unknown directive: %s", toString(t.token));
    };

    std::optional<Token> tok;

    while (true) {
      tok = nextToken(TokenOptional);
      if (!tok.has_value())
        break;

      switch (tok->token[0]) {
      case 'A':
        if (tok->token == "AttributeBegin")
          target->AttributeBegin(tok->loc);
        else if (tok->token == "AttributeEnd")
          target->AttributeEnd(tok->loc);
        else if (tok->token == "Attribute")
          basicParamListEntrypoint(&ParserTarget::Attribute, tok->loc);
        else if (tok->token == "ActiveTransform") {
          Token a = *nextToken(TokenRequired);
          if (a.token == "All")
            target->ActiveTransformAll(tok->loc);
          else if (a.token == "EndTime")
            target->ActiveTransformEndTime(tok->loc);
          else if (a.token == "StartTime")
            target->ActiveTransformStartTime(tok->loc);
          else
            syntaxError(*tok);
        }
        else if (tok->token == "AreaLightSource")
          basicParamListEntrypoint(&ParserTarget::AreaLightSource, tok->loc);
        else if (tok->token == "Accelerator")
          basicParamListEntrypoint(&ParserTarget::Accelerator, tok->loc);
        else
          syntaxError(*tok);
        break;

      case 'C':
        if (tok->token == "ConcatTransform") {
          if (nextToken(TokenRequired)->token != "[")
            syntaxError(*tok);
          Float m[16];
          for (int i = 0; i < 16; ++i)
            m[i] = parseFloat(*nextToken(TokenRequired));
          if (nextToken(TokenRequired)->token != "]")
            syntaxError(*tok);
          target->ConcatTransform(m, tok->loc);
        }
        else if (tok->token == "CoordinateSystem") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          target->CoordinateSystem(toString(n), tok->loc);
        }
        else if (tok->token == "CoordSysTransform") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          target->CoordSysTransform(toString(n), tok->loc);
        }
        else if (tok->token == "ColorSpace") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          target->ColorSpace(toString(n), tok->loc);
        }
        else if (tok->token == "Camera")
          basicParamListEntrypoint(&ParserTarget::Camera, tok->loc);
        else
          syntaxError(*tok);
        break;

      case 'F':
        if (tok->token == "Film")
          basicParamListEntrypoint(&ParserTarget::Film, tok->loc);
        else
          syntaxError(*tok);
        break;

      case 'I':
        if (tok->token == "Integrator")
          basicParamListEntrypoint(&ParserTarget::Integrator, tok->loc);
        else if (tok->token == "Include") {
          Token filenameToken = *nextToken(TokenRequired);
          std::string filename = toString(dequoteString(filenameToken));
          if (true) {
            filename = ResolveFilename(filename);
            std::unique_ptr<Tokenizer> tinc =
              Tokenizer::CreateFromFile(filename, parseError);
            if (tinc) {
              LOG_VERBOSE("Started parsing %s",
                std::string(tinc->loc.filename.begin(),
                  tinc->loc.filename.end()));
              fileStack.push_back(std::move(tinc));
            }
          }
        }
        else if (tok->token == "Import") {
          //Token filenameToken = *nextToken(TokenRequired);
          //std::string filename = toString(dequoteString(filenameToken));
          //if (true) {
          //  BasicSceneBuilder* builder =
          //    dynamic_cast<BasicSceneBuilder*>(target);
          //  CHECK(builder);

          //  if (builder->currentBlock !=
          //    BasicSceneBuilder::BlockState::WorldBlock)
          //    ErrorExit(&tok->loc, "Import statement only allowed inside world "
          //      "definition block.");

          //  filename = ResolveFilename(filename);
          //  BasicSceneBuilder* importBuilder = builder->CopyForImport();

          //  if (RunningThreads() == 1) {
          //    std::unique_ptr<Tokenizer> timport =
          //      Tokenizer::CreateFromFile(filename, parseError);
          //    if (timport)
          //      parse(importBuilder, std::move(timport));
          //    builder->MergeImported(importBuilder);
          //  }
          //  else {
          //    auto job = [=](std::string filename) {
          //      Timer timer;
          //      std::unique_ptr<Tokenizer> timport =
          //        Tokenizer::CreateFromFile(filename, parseError);
          //      if (timport)
          //        parse(importBuilder, std::move(timport));
          //      LOG_VERBOSE("Elapsed time to parse \"%s\": %.2fs", filename,
          //        timer.ElapsedSeconds());
          //      return 0;
          //    };
          //    AsyncJob<int>* jobFinished = RunAsync(job, filename);
          //    imports.push_back(std::make_pair(jobFinished, importBuilder));
          //  }
          //}
        }
        else if (tok->token == "Identity")
          target->Identity(tok->loc);
        else
          syntaxError(*tok);
        break;

      case 'L':
        if (tok->token == "LightSource")
          basicParamListEntrypoint(&ParserTarget::LightSource, tok->loc);
        else if (tok->token == "LookAt") {
          Float v[9];
          for (int i = 0; i < 9; ++i)
            v[i] = parseFloat(*nextToken(TokenRequired));
          target->LookAt(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
            tok->loc);
        }
        else
          syntaxError(*tok);
        break;

      case 'M':
        if (tok->token == "MakeNamedMaterial")
          basicParamListEntrypoint(&ParserTarget::MakeNamedMaterial, tok->loc);
        else if (tok->token == "MakeNamedMedium")
          basicParamListEntrypoint(&ParserTarget::MakeNamedMedium, tok->loc);
        else if (tok->token == "Material")
          basicParamListEntrypoint(&ParserTarget::Material, tok->loc);
        else if (tok->token == "MediumInterface") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          std::string names[2];
          names[0] = toString(n);

          // Check for optional second parameter
          pstd::optional<Token> second = nextToken(TokenOptional);
          if (second.has_value()) {
            if (isQuotedString(second->token))
              names[1] = toString(dequoteString(*second));
            else {
              unget(*second);
              names[1] = names[0];
            }
          }
          else
            names[1] = names[0];

          target->MediumInterface(names[0], names[1], tok->loc);
        }
        else
          syntaxError(*tok);
        break;

      case 'N':
        if (tok->token == "NamedMaterial") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          target->NamedMaterial(toString(n), tok->loc);
        }
        else
          syntaxError(*tok);
        break;

      case 'O':
        if (tok->token == "ObjectBegin") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          target->ObjectBegin(toString(n), tok->loc);
        }
        else if (tok->token == "ObjectEnd")
          target->ObjectEnd(tok->loc);
        else if (tok->token == "ObjectInstance") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          target->ObjectInstance(toString(n), tok->loc);
        }
        else if (tok->token == "Option") {
          std::string name = toString(dequoteString(*nextToken(TokenRequired)));
          std::string value = toString(nextToken(TokenRequired)->token);
          target->Option(name, value, tok->loc);
        }
        else
          syntaxError(*tok);
        break;

      case 'P':
        if (tok->token == "PixelFilter")
          basicParamListEntrypoint(&ParserTarget::PixelFilter, tok->loc);
        else
          syntaxError(*tok);
        break;

      case 'R':
        if (tok->token == "ReverseOrientation")
          target->ReverseOrientation(tok->loc);
        else if (tok->token == "Rotate") {
          Float v[4];
          for (int i = 0; i < 4; ++i)
            v[i] = parseFloat(*nextToken(TokenRequired));
          target->Rotate(v[0], v[1], v[2], v[3], tok->loc);
        }
        else
          syntaxError(*tok);
        break;

      case 'S':
        if (tok->token == "Shape")
          basicParamListEntrypoint(&ParserTarget::Shape, tok->loc);
        else if (tok->token == "Sampler")
          basicParamListEntrypoint(&ParserTarget::Sampler, tok->loc);
        else if (tok->token == "Scale") {
          Float v[3];
          for (int i = 0; i < 3; ++i)
            v[i] = parseFloat(*nextToken(TokenRequired));
          target->Scale(v[0], v[1], v[2], tok->loc);
        }
        else
          syntaxError(*tok);
        break;

      case 'T':
        if (tok->token == "TransformBegin") {
          if (true) {
            if (!warnedTransformBeginEndDeprecated) {
              Warning(&tok->loc, "TransformBegin/End are deprecated and should "
                "be replaced with AttributeBegin/End");
              warnedTransformBeginEndDeprecated = true;
            }
            target->AttributeBegin(tok->loc);
          }
        }
        else if (tok->token == "TransformEnd") {
          target->AttributeEnd(tok->loc);
        }
        else if (tok->token == "Transform") {
          if (nextToken(TokenRequired)->token != "[")
            syntaxError(*tok);
          Float m[16];
          for (int i = 0; i < 16; ++i)
            m[i] = parseFloat(*nextToken(TokenRequired));
          if (nextToken(TokenRequired)->token != "]")
            syntaxError(*tok);
          target->Transform(m, tok->loc);
        }
        else if (tok->token == "Translate") {
          Float v[3];
          for (int i = 0; i < 3; ++i)
            v[i] = parseFloat(*nextToken(TokenRequired));
          target->Translate(v[0], v[1], v[2], tok->loc);
        }
        else if (tok->token == "TransformTimes") {
          Float v[2];
          for (int i = 0; i < 2; ++i)
            v[i] = parseFloat(*nextToken(TokenRequired));
          target->TransformTimes(v[0], v[1], tok->loc);
        }
        else if (tok->token == "Texture") {
          std::string_view n = dequoteString(*nextToken(TokenRequired));
          std::string name = toString(n);
          n = dequoteString(*nextToken(TokenRequired));
          std::string type = toString(n);

          Token t = *nextToken(TokenRequired);
          std::string_view dequoted = dequoteString(t);
          std::string texName = toString(dequoted);
          ParsedParameterVector params = parseParameters(
            nextToken, unget, false, [&](const Token& t, const char* msg) {
              std::string token = toString(t.token);
          std::string str = StringPrintf("%s: %s", token, msg);
          parseError(str.c_str(), &t.loc);
            });

          target->Texture(name, type, texName, std::move(params), tok->loc);
        }
        else
          syntaxError(*tok);
        break;

      case 'W':
        if (tok->token == "WorldBegin")
          target->WorldBegin(tok->loc);
        else if (tok->token == "WorldEnd")
          ;  // just swallow it
        else
          syntaxError(*tok);
        break;

      default:
        syntaxError(*tok);
      }
    }

    //for (auto& import : imports) {
    //  import.first->Wait();

    //  BasicSceneBuilder* builder = dynamic_cast<BasicSceneBuilder*>(target);
    //  CHECK(builder);
    //  builder->MergeImported(import.second);
    //  // HACK: let import.second leak so that its TransformCache isn't deallocated...
    //}
  }

  void ParseString(ParserTarget* target, std::string str) {
    auto tokError = [](const char* msg, const FileLoc* loc) {
      ErrorExit(loc, "%s", msg);
    };
    std::unique_ptr<Tokenizer> t = Tokenizer::CreateFromString(std::move(str), tokError);
    if (!t)
      return;
    parse(target, std::move(t));

    target->EndOfFiles();
  }

  // SquareMatrix Definition
  template <int N>
  class SquareMatrix {
  public:
    // SquareMatrix Public Methods
    PBRT_CPU_GPU
      static SquareMatrix Zero() {
      SquareMatrix m;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          m.m[i][j] = 0;
      return m;
    }

    PBRT_CPU_GPU
      SquareMatrix() {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          m[i][j] = (i == j) ? 1 : 0;
    }
    PBRT_CPU_GPU
      SquareMatrix(const Float mat[N][N]) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          m[i][j] = mat[i][j];
    }
    PBRT_CPU_GPU
      SquareMatrix(pstd::span<const Float> t);
    template <typename... Args>
    PBRT_CPU_GPU SquareMatrix(Float v, Args... args) {
      static_assert(1 + sizeof...(Args) == N * N,
        "Incorrect number of values provided to SquareMatrix constructor");
      init<N>(m, 0, 0, v, args...);
    }
    template <typename... Args>
    PBRT_CPU_GPU static SquareMatrix Diag(Float v, Args... args) {
      static_assert(1 + sizeof...(Args) == N,
        "Incorrect number of values provided to SquareMatrix::Diag");
      SquareMatrix m;
      initDiag<N>(m.m, 0, v, args...);
      return m;
    }

    PBRT_CPU_GPU
      SquareMatrix operator+(const SquareMatrix& m) const {
      SquareMatrix r = *this;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          r.m[i][j] += m.m[i][j];
      return r;
    }

    PBRT_CPU_GPU
      SquareMatrix operator*(Float s) const {
      SquareMatrix r = *this;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          r.m[i][j] *= s;
      return r;
    }
    PBRT_CPU_GPU
      SquareMatrix operator/(Float s) const {
      DCHECK_NE(s, 0);
      SquareMatrix r = *this;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          r.m[i][j] /= s;
      return r;
    }

    PBRT_CPU_GPU
      bool operator==(const SquareMatrix<N>& m2) const {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          if (m[i][j] != m2.m[i][j])
            return false;
      return true;
    }

    PBRT_CPU_GPU
      bool operator!=(const SquareMatrix<N>& m2) const {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          if (m[i][j] != m2.m[i][j])
            return true;
      return false;
    }

    PBRT_CPU_GPU
      bool operator<(const SquareMatrix<N>& m2) const {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          if (m[i][j] < m2.m[i][j])
            return true;
          if (m[i][j] > m2.m[i][j])
            return false;
        }
      return false;
    }

    PBRT_CPU_GPU
      bool IsIdentity() const;

    std::string ToString() const;

    PBRT_CPU_GPU
      pstd::span<const Float> operator[](int i) const { return m[i]; }
    PBRT_CPU_GPU
      pstd::span<Float> operator[](int i) { return pstd::span<Float>(m[i]); }

    Float m[N][N];
  };

  // SquareMatrix Inline Methods
  template <int N>
  PBRT_CPU_GPU inline bool SquareMatrix<N>::IsIdentity() const {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        if (i == j) {
          if (m[i][j] != 1)
            return false;
        }
        else if (m[i][j] != 0)
          return false;
      }
    return true;
  }

  // SquareMatrix Inline Functions
  template <int N>
  PBRT_CPU_GPU inline SquareMatrix<N> operator*(Float s, const SquareMatrix<N>& m) {
    return m * s;
  }

  template <typename Tresult, int N, typename T>
  PBRT_CPU_GPU inline Tresult Mul(const SquareMatrix<N>& m, const T& v) {
    Tresult result;
    for (int i = 0; i < N; ++i) {
      result[i] = 0;
      for (int j = 0; j < N; ++j)
        result[i] += m[i][j] * v[j];
    }
    return result;
  }

  template <int N>
  PBRT_CPU_GPU Float Determinant(const SquareMatrix<N>& m);

  template <typename T>
  inline PBRT_CPU_GPU T FMA(T a, T b, T c) {
    return a * b + c;
  }

  // CompensatedFloat Definition
  struct CompensatedFloat {
  public:
    // CompensatedFloat Public Methods
    PBRT_CPU_GPU
      CompensatedFloat(Float v, Float err = 0) : v(v), err(err) {}
    PBRT_CPU_GPU
      explicit operator float() const { return v + err; }
    PBRT_CPU_GPU
      explicit operator double() const { return double(v) + double(err); }
    std::string ToString() const;

    Float v, err;
  };

  // InnerProduct Helper Functions
  template <typename Float>
  PBRT_CPU_GPU inline CompensatedFloat InnerProduct(Float a, Float b) {
    return TwoProd(a, b);
  }

  template <typename Ta, typename Tb, typename Tc, typename Td>
  PBRT_CPU_GPU inline auto DifferenceOfProducts(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto differenceOfProducts = FMA(a, b, -cd);
    auto error = FMA(-c, d, cd);
    return differenceOfProducts + error;
  }

  template <>
  PBRT_CPU_GPU inline Float Determinant(const SquareMatrix<3>& m) {
    Float minor12 = DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    Float minor02 = DifferenceOfProducts(m[1][0], m[2][2], m[1][2], m[2][0]);
    Float minor01 = DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    return FMA(m[0][2], minor01,
      DifferenceOfProducts(m[0][0], minor12, m[0][1], minor02));
  }

  template <int N>
  PBRT_CPU_GPU inline SquareMatrix<N> Transpose(const SquareMatrix<N>& m);
  template <int N>
  PBRT_CPU_GPU pstd::optional<SquareMatrix<N>> Inverse(const SquareMatrix<N>&);

  template <int N>
  PBRT_CPU_GPU inline SquareMatrix<N> Transpose(const SquareMatrix<N>& m) {
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        r[i][j] = m[j][i];
    return r;
  }

  template <>
  PBRT_CPU_GPU inline pstd::optional<SquareMatrix<3>> Inverse(const SquareMatrix<3>& m) {
    Float det = Determinant(m);
    if (det == 0)
      return {};
    Float invDet = 1 / det;

    SquareMatrix<3> r;

    r[0][0] = invDet * DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    r[1][0] = invDet * DifferenceOfProducts(m[1][2], m[2][0], m[1][0], m[2][2]);
    r[2][0] = invDet * DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    r[0][1] = invDet * DifferenceOfProducts(m[0][2], m[2][1], m[0][1], m[2][2]);
    r[1][1] = invDet * DifferenceOfProducts(m[0][0], m[2][2], m[0][2], m[2][0]);
    r[2][1] = invDet * DifferenceOfProducts(m[0][1], m[2][0], m[0][0], m[2][1]);
    r[0][2] = invDet * DifferenceOfProducts(m[0][1], m[1][2], m[0][2], m[1][1]);
    r[1][2] = invDet * DifferenceOfProducts(m[0][2], m[1][0], m[0][0], m[1][2]);
    r[2][2] = invDet * DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);

    return r;
  }

  template <int N, typename T>
  PBRT_CPU_GPU inline T operator*(const SquareMatrix<N>& m, const T& v) {
    return Mul<T>(m, v);
  }

  PBRT_CPU_GPU inline CompensatedFloat TwoProd(Float a, Float b) {
    Float ab = a * b;
    return { ab, FMA(a, b, -ab) };
  }

  PBRT_CPU_GPU inline CompensatedFloat TwoSum(Float a, Float b) {
    Float s = a + b, delta = s - a;
    return { s, (a - (s - delta)) + (b - delta) };
  }

  namespace internal {
    // InnerProduct Helper Functions
    template <typename Float>
    PBRT_CPU_GPU inline CompensatedFloat InnerProduct(Float a, Float b) {
      return TwoProd(a, b);
    }

    // Accurate dot products with FMA: Graillat et al.,
    // https://www-pequan.lip6.fr/~graillat/papers/posterRNC7.pdf
    //
    // Accurate summation, dot product and polynomial evaluation in complex
    // floating point arithmetic, Graillat and Menissier-Morain.
    template <typename Float, typename... T>
    PBRT_CPU_GPU inline CompensatedFloat InnerProduct(Float a, Float b, T... terms) {
      CompensatedFloat ab = TwoProd(a, b);
      CompensatedFloat tp = InnerProduct(terms...);
      CompensatedFloat sum = TwoSum(ab.v, tp.v);
      return { sum.v, ab.err + (tp.err + sum.err) };
    }
  }

  template <typename... T>
  PBRT_CPU_GPU inline std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>...>, Float>
    InnerProduct(T... terms) {
    CompensatedFloat ip = internal::InnerProduct(terms...);
    return Float(ip);
  }

  template <>
  PBRT_CPU_GPU inline SquareMatrix<3> operator*(const SquareMatrix<3>& m1,
    const SquareMatrix<3>& m2) {
    SquareMatrix<3> r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r[i][j] =
        InnerProduct(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2], m2[2][j]);
    return r;
  }

  template <int N>
  PBRT_CPU_GPU inline SquareMatrix<N> operator*(const SquareMatrix<N>& m1,
    const SquareMatrix<N>& m2) {
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        r[i][j] = 0;
        for (int k = 0; k < N; ++k)
          r[i][j] = FMA(m1[i][k], m2[k][j], r[i][j]);
      }
    return r;
  }

  template <int N>
  PBRT_CPU_GPU inline SquareMatrix<N>::SquareMatrix(pstd::span<const Float> t) {
    CHECK_EQ(N * N, t.size());
    for (int i = 0; i < N * N; ++i)
      m[i / N][i % N] = t[i];
  }

  template <int N>
  PBRT_CPU_GPU SquareMatrix<N> operator*(const SquareMatrix<N>& m1,
    const SquareMatrix<N>& m2);

  template <>
  PBRT_CPU_GPU inline Float Determinant(const SquareMatrix<1>& m) {
    return m[0][0];
  }

  template <>
  PBRT_CPU_GPU inline Float Determinant(const SquareMatrix<2>& m) {
    return DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);
  }

  template <>
  PBRT_CPU_GPU inline Float Determinant(const SquareMatrix<4>& m) {
    Float s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
    Float s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
    Float s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);

    Float s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
    Float s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
    Float s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

    Float c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
    Float c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
    Float c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);

    Float c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
    Float c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
    Float c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

    return (DifferenceOfProducts(s0, c5, s1, c4) + DifferenceOfProducts(s2, c3, -s3, c2) +
      DifferenceOfProducts(s5, c0, s4, c1));
  }

  template <>
  PBRT_CPU_GPU inline pstd::optional<SquareMatrix<4>> Inverse(const SquareMatrix<4>& m) {
    // Via: https://github.com/google/ion/blob/master/ion/math/matrixutils.cc,
    // (c) Google, Apache license.

    // For 4x4 do not compute the adjugate as the transpose of the cofactor
    // matrix, because this results in extra work. Several calculations can be
    // shared across the sub-determinants.
    //
    // This approach is explained in David Eberly's Geometric Tools book,
    // excerpted here:
    //   http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    Float s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
    Float s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
    Float s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);

    Float s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
    Float s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
    Float s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

    Float c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
    Float c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
    Float c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);

    Float c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
    Float c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
    Float c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

    Float determinant = InnerProduct(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (determinant == 0)
      return {};
    Float s = 1 / determinant;

    Float inv[4][4] = { {s * InnerProduct(m[1][1], c5, m[1][3], c3, -m[1][2], c4),
                        s * InnerProduct(-m[0][1], c5, m[0][2], c4, -m[0][3], c3),
                        s * InnerProduct(m[3][1], s5, m[3][3], s3, -m[3][2], s4),
                        s * InnerProduct(-m[2][1], s5, m[2][2], s4, -m[2][3], s3)},

                       {s * InnerProduct(-m[1][0], c5, m[1][2], c2, -m[1][3], c1),
                        s * InnerProduct(m[0][0], c5, m[0][3], c1, -m[0][2], c2),
                        s * InnerProduct(-m[3][0], s5, m[3][2], s2, -m[3][3], s1),
                        s * InnerProduct(m[2][0], s5, m[2][3], s1, -m[2][2], s2)},

                       {s * InnerProduct(m[1][0], c4, m[1][3], c0, -m[1][1], c2),
                        s * InnerProduct(-m[0][0], c4, m[0][1], c2, -m[0][3], c0),
                        s * InnerProduct(m[3][0], s4, m[3][3], s0, -m[3][1], s2),
                        s * InnerProduct(-m[2][0], s4, m[2][1], s2, -m[2][3], s0)},

                       {s * InnerProduct(-m[1][0], c3, m[1][1], c1, -m[1][2], c0),
                        s * InnerProduct(m[0][0], c3, m[0][2], c0, -m[0][1], c1),
                        s * InnerProduct(-m[3][0], s3, m[3][1], s1, -m[3][2], s0),
                        s * InnerProduct(m[2][0], s3, m[2][2], s0, -m[2][1], s1)} };

    return SquareMatrix<4>(inv);
  }

  template <int N>
  PBRT_CPU_GPU SquareMatrix<N> InvertOrExit(const SquareMatrix<N>& m) {
    pstd::optional<SquareMatrix<N>> inv = Inverse(m);
    CHECK(inv.has_value());
    return *inv;
  }

  template <int N>
  PBRT_CPU_GPU inline Float Determinant(const SquareMatrix<N>& m) {
    SquareMatrix<N - 1> sub;
    Float det = 0;
    // Inefficient, but we don't currently use N>4 anyway..
    for (int i = 0; i < N; ++i) {
      // Sub-matrix without row 0 and column i
      for (int j = 0; j < N - 1; ++j)
        for (int k = 0; k < N - 1; ++k)
          sub[j][k] = m[j + 1][k < i ? k : k + 1];

      Float sign = (i & 1) ? -1 : 1;
      det += sign * m[0][i] * Determinant(sub);
    }
    return det;
  }

  // Transform Definition
  struct Transform {
    Transform() = default;
    Transform(const SquareMatrix<4>& m) : m(m) {
      pstd::optional<SquareMatrix<4>> inv = Inverse(m);
      if (inv)
        mInv = *inv;
      else {
        // Initialize _mInv_ with not-a-number values
        Float NaN = std::numeric_limits<Float>::has_signaling_NaN
          ? std::numeric_limits<Float>::signaling_NaN()
          : std::numeric_limits<Float>::quiet_NaN();
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j)
            mInv[i][j] = NaN;
      }
    }

    PBRT_CPU_GPU
      Transform(const Float mat[4][4]) : Transform(SquareMatrix<4>(mat)) {}

    PBRT_CPU_GPU
      Transform(const SquareMatrix<4>& m, const SquareMatrix<4>& mInv) : m(m), mInv(mInv) {}

    PBRT_CPU_GPU
      const SquareMatrix<4>& GetMatrix() const { return m; }
    PBRT_CPU_GPU
      const SquareMatrix<4>& GetInverseMatrix() const { return mInv; }

    PBRT_CPU_GPU
      bool operator==(const Transform& t) const { return t.m == m; }
    PBRT_CPU_GPU
      bool operator!=(const Transform& t) const { return t.m != m; }
    PBRT_CPU_GPU
      bool IsIdentity() const { return m.IsIdentity(); }

    PBRT_CPU_GPU Transform operator*(const Transform& t2) const {
      return Transform(m * t2.m, t2.mInv * mInv);
    }

    // Transform Private Members
    SquareMatrix<4> m, mInv;
  };

  // General case
  template <int N>
  PBRT_CPU_GPU pstd::optional<SquareMatrix<N>> Inverse(const SquareMatrix<N>& m) {
    int indxc[N], indxr[N];
    int ipiv[N] = { 0 };
    Float minv[N][N];
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        minv[i][j] = m[i][j];
    for (int i = 0; i < N; i++) {
      int irow = 0, icol = 0;
      Float big = 0.f;
      // Choose pivot
      for (int j = 0; j < N; j++) {
        if (ipiv[j] != 1) {
          for (int k = 0; k < N; k++) {
            if (ipiv[k] == 0) {
              if (std::abs(minv[j][k]) >= big) {
                big = std::abs(minv[j][k]);
                irow = j;
                icol = k;
              }
            }
            else if (ipiv[k] > 1)
              return {};  // singular
          }
        }
      }
      ++ipiv[icol];
      // Swap rows _irow_ and _icol_ for pivot
      if (irow != icol) {
        for (int k = 0; k < N; ++k)
          pstd::swap(minv[irow][k], minv[icol][k]);
      }
      indxr[i] = irow;
      indxc[i] = icol;
      if (minv[icol][icol] == 0.f)
        return {};  // singular

      // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
      Float pivinv = 1. / minv[icol][icol];
      minv[icol][icol] = 1.;
      for (int j = 0; j < N; j++)
        minv[icol][j] *= pivinv;

      // Subtract this row from others to zero out their columns
      for (int j = 0; j < N; j++) {
        if (j != icol) {
          Float save = minv[j][icol];
          minv[j][icol] = 0;
          for (int k = 0; k < N; k++)
            minv[j][k] = FMA(-minv[icol][k], save, minv[j][k]);
        }
      }
    }
    // Swap columns to reflect permutation
    for (int j = N - 1; j >= 0; j--) {
      if (indxr[j] != indxc[j]) {
        for (int k = 0; k < N; k++)
          pstd::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
      }
    }
    return SquareMatrix<N>(minv);
  }

  // Transform Inline Functions
  PBRT_CPU_GPU inline Transform Inverse(const Transform& t) {
    return Transform(t.GetInverseMatrix(), t.GetMatrix());
  }

  // MaxTransforms Definition
  constexpr int MaxTransforms = 2;

  // TransformSet Definition
  struct TransformSet {
    // TransformSet Public Methods
    Transform& operator[](int i) {
      CHECK_GE(i, 0);
      CHECK_LT(i, MaxTransforms);
      return t[i];
    }
    const Transform& operator[](int i) const {
      CHECK_GE(i, 0);
      CHECK_LT(i, MaxTransforms);
      return t[i];
    }
    friend TransformSet Inverse(const TransformSet& ts) {
      TransformSet tInv;
      for (int i = 0; i < MaxTransforms; ++i)
        tInv.t[i] = Inverse(ts.t[i]);
      return tInv;
    }
    bool IsAnimated() const {
      for (int i = 0; i < MaxTransforms - 1; ++i)
        if (t[i] != t[i + 1])
          return true;
      return false;
    }

  private:
    Transform t[MaxTransforms];
  };

  static constexpr int EndTransformBits = 1 << 1;
  static constexpr int AllTransformsBits = (1 << MaxTransforms) - 1;

  //// InternCache Definition
  //template <typename T, typename Hash = std::hash<T>>
  //class InternCache {
  //public:
  //  // InternCache Public Methods
  //  InternCache(Allocator alloc = {})
  //    : hashTable(256, alloc),
  //    bufferResource(alloc.resource()),
  //    itemAlloc(&bufferResource) {}

  //  template <typename F>
  //  const T* Lookup(const T& item, F create) {
  //    size_t offset = Hash()(item) % hashTable.size();
  //    int step = 1;
  //    mutex.lock_shared();
  //    while (true) {
  //      // Check _hashTable[offset]_ for provided item
  //      if (!hashTable[offset]) {
  //        // Insert item into open hash table entry
  //        mutex.unlock_shared();
  //        mutex.lock();
  //        // Double check that another thread hasn't inserted _item_
  //        size_t offset = Hash()(item) % hashTable.size();
  //        int step = 1;
  //        while (true) {
  //          if (!hashTable[offset])
  //            // fine--it's definitely not there
  //            break;
  //          else if (*hashTable[offset] == item) {
  //            // Another thread inserted it
  //            const T* ret = hashTable[offset];
  //            mutex.unlock();
  //            return ret;
  //          }
  //          else {
  //            // collision
  //            offset += step;
  //            ++step;
  //            offset %= hashTable.size();
  //          }
  //        }

  //        // Grow the hash table if needed
  //        if (4 * nEntries > hashTable.size()) {
  //          pstd::vector<const T*> newHash(2 * hashTable.size(),
  //            hashTable.get_allocator());
  //          for (const T* ptr : hashTable)
  //            if (ptr)
  //              Insert(ptr, &newHash);

  //          hashTable.swap(newHash);
  //        }

  //        // Allocate new hash table entry and add it to the hash table
  //        ++nEntries;
  //        T* newPtr = create(itemAlloc, item);
  //        Insert(newPtr, &hashTable);
  //        mutex.unlock();
  //        return newPtr;

  //      }
  //      else if (*hashTable[offset] == item) {
  //        // Return pointer for found _item_ in hash table
  //        const T* ret = hashTable[offset];
  //        mutex.unlock_shared();
  //        return ret;

  //      }
  //      else {
  //        // Advance _offset_ after hash table collision
  //        offset += step;
  //        ++step;
  //        offset %= hashTable.size();
  //      }
  //    }
  //  }

  //  const T* Lookup(const T& item) {
  //    return Lookup(item, [](Allocator alloc, const T& item) {
  //      return alloc.new_object<T>(item);
  //      });
  //  }

  //  size_t size() const { return nEntries; }
  //  size_t capacity() const { return hashTable.size(); }

  //private:
  //  // InternCache Private Methods
  //  void Insert(const T* ptr, pstd::vector<const T*>* table) {
  //    size_t offset = Hash()(*ptr) % table->size();
  //    int step = 1;
  //    // Advance _offset_ to next free entry in hash table
  //    while ((*table)[offset]) {
  //      offset += step;
  //      ++step;
  //      offset %= table->size();
  //    }

  //    (*table)[offset] = ptr;
  //  }

  //  // InternCache Private Members
  //  pstd::pmr::monotonic_buffer_resource bufferResource;
  //  Allocator itemAlloc;
  //  size_t nEntries = 0;
  //  pstd::vector<const T*> hashTable;
  //  std::shared_mutex mutex;
  //};

  // BasicSceneBuilder Definition
  class BasicSceneBuilder : public ParserTarget {
  public:
    // BasicSceneBuilder Public Methods
    BasicSceneBuilder(BasicScene* scene);
    void Option(const std::string& name, const std::string& value, FileLoc loc);
    void Identity(FileLoc loc);
    void Translate(Float dx, Float dy, Float dz, FileLoc loc);
    void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc);
    void Scale(Float sx, Float sy, Float sz, FileLoc loc);
    void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux,
      Float uy, Float uz, FileLoc loc);
    void ConcatTransform(Float transform[16], FileLoc loc);
    void Transform(Float transform[16], FileLoc loc);
    void CoordinateSystem(const std::string&, FileLoc loc);
    void CoordSysTransform(const std::string&, FileLoc loc);
    void ActiveTransformAll(FileLoc loc);
    void ActiveTransformEndTime(FileLoc loc);
    void ActiveTransformStartTime(FileLoc loc);
    void TransformTimes(Float start, Float end, FileLoc loc);
    void ColorSpace(const std::string& n, FileLoc loc);
    void PixelFilter(const std::string& name, ParsedParameterVector params, FileLoc loc);
    void Film(const std::string& type, ParsedParameterVector params, FileLoc loc);
    void Sampler(const std::string& name, ParsedParameterVector params, FileLoc loc);
    void Accelerator(const std::string& name, ParsedParameterVector params, FileLoc loc);
    void Integrator(const std::string& name, ParsedParameterVector params, FileLoc loc);
    void Camera(const std::string&, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMedium(const std::string& name, ParsedParameterVector params,
      FileLoc loc);
    void MediumInterface(const std::string& insideName, const std::string& outsideName,
      FileLoc loc);
    void WorldBegin(FileLoc loc);
    void AttributeBegin(FileLoc loc);
    void AttributeEnd(FileLoc loc);
    void Attribute(const std::string& target, ParsedParameterVector params, FileLoc loc);
    void Texture(const std::string& name, const std::string& type,
      const std::string& texname, ParsedParameterVector params, FileLoc loc);
    void Material(const std::string& name, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMaterial(const std::string& name, ParsedParameterVector params,
      FileLoc loc);
    void NamedMaterial(const std::string& name, FileLoc loc);
    void LightSource(const std::string& name, ParsedParameterVector params, FileLoc loc);
    void AreaLightSource(const std::string& name, ParsedParameterVector params,
      FileLoc loc);
    void Shape(const std::string& name, ParsedParameterVector params, FileLoc loc);
    void ReverseOrientation(FileLoc loc);
    void ObjectBegin(const std::string& name, FileLoc loc);
    void ObjectEnd(FileLoc loc);
    void ObjectInstance(const std::string& name, FileLoc loc);

    void EndOfFiles();

    BasicSceneBuilder* CopyForImport();
    void MergeImported(BasicSceneBuilder*);

    std::string ToString() const;

    // BasicSceneBuilder::GraphicsState Definition
    struct GraphicsState {
      // GraphicsState Public Methods
      GraphicsState();

      template <typename F>
      void ForActiveTransforms(F func) {
        for (int i = 0; i < MaxTransforms; ++i)
          if (activeTransformBits & (1 << i))
            ctm[i] = func(ctm[i]);
      }

      // GraphicsState Public Members
      std::string currentInsideMedium, currentOutsideMedium;

      int currentMaterialIndex = 0;
      std::string currentMaterialName;

      std::string areaLightName;
      ParameterDictionary areaLightParams;
      FileLoc areaLightLoc;

      ParsedParameterVector shapeAttributes;
      ParsedParameterVector lightAttributes;
      ParsedParameterVector materialAttributes;
      ParsedParameterVector mediumAttributes;
      ParsedParameterVector textureAttributes;
      bool reverseOrientation = false;
      //const RGBColorSpace* colorSpace = RGBColorSpace::sRGB;
      TransformSet ctm;
      uint32_t activeTransformBits = AllTransformsBits;
      Float transformStartTime = 0, transformEndTime = 1;
    };

    //friend void parse(ParserTarget* scene, std::unique_ptr<Tokenizer> t);
    // BasicSceneBuilder Private Methods
    class Transform RenderFromObject(int index) const {
      return tiny_pbrt_loader::Transform((renderFromWorld * graphicsState.ctm[index]).GetMatrix());
    }

    //AnimatedTransform RenderFromObject() const {
    //  return { RenderFromObject(0), graphicsState.transformStartTime,
    //          RenderFromObject(1), graphicsState.transformEndTime };
    //}

    //bool CTMIsAnimated() const { return graphicsState.ctm.IsAnimated(); }

    // BasicSceneBuilder Private Members
    BasicScene* scene;
    enum class BlockState { OptionsBlock, WorldBlock };
    BlockState currentBlock = BlockState::OptionsBlock;
    GraphicsState graphicsState;
    //static constexpr int StartTransformBits = 1 << 0;
    //static constexpr int EndTransformBits = 1 << 1;
    //static constexpr int AllTransformsBits = (1 << MaxTransforms) - 1;
    std::map<std::string, TransformSet> namedCoordinateSystems;
    class Transform renderFromWorld;
    //InternCache<class Transform> transformCache;
    std::vector<GraphicsState> pushedGraphicsStates;
    std::vector<std::pair<char, FileLoc>> pushStack;  // 'a': attribute, 'o': object
    //struct ActiveInstanceDefinition {
    //  ActiveInstanceDefinition(std::string name, FileLoc loc) : entity(name, loc) {}

    //  std::mutex mutex;
    //  std::atomic<int> activeImports{ 1 };
    //  InstanceDefinitionSceneEntity entity;
    //  ActiveInstanceDefinition* parent = nullptr;
    //};
    //ActiveInstanceDefinition* activeInstanceDefinition = nullptr;

    //// Buffer these both to avoid mutex contention and so that they are
    //// consistently ordered across runs.
    std::vector<ShapeSceneEntity> shapes;
    std::vector<InstanceSceneEntity> instanceUses;

    std::set<std::string> namedMaterialNames, mediumNames;
    //std::set<std::string> floatTextureNames, spectrumTextureNames, instanceNames;
    //int currentMaterialIndex = 0, currentLightIndex = -1;
    SceneEntity sampler;
    SceneEntity film, integrator, filter, accelerator;
    CameraSceneEntity camera;
  };

  BasicSceneBuilder::GraphicsState::GraphicsState() {
    currentMaterialIndex = 0;
  }

  template <int &...ExplicitArgumentBarrier, typename T>
  PBRT_CPU_GPU inline constexpr std::span<T> MakeSpan(T* ptr, size_t size) noexcept {
    return std::span<T>(ptr, size);
  }

  template <int &...ExplicitArgumentBarrier, typename T>
  PBRT_CPU_GPU inline std::span<T> MakeSpan(T* begin, T* end) noexcept {
    return std::span<T>(begin, end - begin);
  }

  template <int &...ExplicitArgumentBarrier, typename T>
  inline std::span<T> MakeSpan(std::vector<T>& v) noexcept {
    return std::span<T>(v.data(), v.size());
  }

  // BasicSceneBuilder Method Definitions
  BasicSceneBuilder::BasicSceneBuilder(BasicScene* scene) : scene(scene) {
    // Set scene defaults
    camera.name = "perspective";
    sampler.name = "zsobol";
    filter.name = "gaussian";
    integrator.name = "volpath";
    accelerator.name = "bvh";

    film.name = "rgb";
    //film.parameters = ParameterDictionary({}, RGBColorSpace::sRGB);

    //ParameterDictionary dict({}, RGBColorSpace::sRGB);
    //currentMaterialIndex = scene->AddMaterial(SceneEntity("diffuse", dict, {}));
  }
  void BasicSceneBuilder::Option(const std::string& name, const std::string& value, FileLoc loc) {

  }
  void BasicSceneBuilder::Identity(FileLoc loc) {

  }
  void BasicSceneBuilder::Translate(Float dx, Float dy, Float dz, FileLoc loc) {

  }
  void BasicSceneBuilder::Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc) {

  }
  void BasicSceneBuilder::Scale(Float sx, Float sy, Float sz, FileLoc loc) {

  }
  void BasicSceneBuilder::LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux,
    Float uy, Float uz, FileLoc loc) {

  }
  PBRT_CPU_GPU inline Transform Transpose(const Transform& t) {
    return Transform(Transpose(t.GetMatrix()), Transpose(t.GetInverseMatrix()));
  }
  void BasicSceneBuilder::ConcatTransform(Float tr[16], FileLoc loc) {
    graphicsState.ForActiveTransforms([=](auto t) {
      return t * Transpose(tiny_pbrt_loader::Transform(SquareMatrix<4>(MakeSpan(tr, 16))));
      });
  }
  void BasicSceneBuilder::Transform(Float tr[16], FileLoc loc) {
    graphicsState.ForActiveTransforms([=](auto t) {
      return Transpose(tiny_pbrt_loader::Transform(SquareMatrix<4>(MakeSpan(tr, 16))));
      });
  }
  void BasicSceneBuilder::CoordinateSystem(const std::string& name, FileLoc loc) {
    namedCoordinateSystems[name] = graphicsState.ctm;
  }
  void BasicSceneBuilder::CoordSysTransform(const std::string&, FileLoc loc) {

  }
  void BasicSceneBuilder::ActiveTransformAll(FileLoc loc) {

  }
  void BasicSceneBuilder::ActiveTransformEndTime(FileLoc loc) {

  }
  void BasicSceneBuilder::ActiveTransformStartTime(FileLoc loc) {

  }
  void BasicSceneBuilder::TransformTimes(Float start, Float end, FileLoc loc) {

  }
  void BasicSceneBuilder::ColorSpace(const std::string& n, FileLoc loc) {

  }
  void BasicSceneBuilder::PixelFilter(const std::string& name, ParsedParameterVector params, FileLoc loc) {

  }
  void BasicSceneBuilder::Film(const std::string& type, ParsedParameterVector params, FileLoc loc) {

  }
  void BasicSceneBuilder::Sampler(const std::string& name, ParsedParameterVector params, FileLoc loc) {

  }
  void BasicSceneBuilder::Accelerator(const std::string& name, ParsedParameterVector params, FileLoc loc) {

  }
  void BasicSceneBuilder::Integrator(const std::string& name, ParsedParameterVector params, FileLoc loc) {

  }
  void BasicSceneBuilder::Camera(const std::string& name, ParsedParameterVector params, FileLoc loc) {
    TransformSet cameraFromWorld = graphicsState.ctm;
    TransformSet worldFromCamera = Inverse(graphicsState.ctm);
    namedCoordinateSystems["camera"] = Inverse(cameraFromWorld);

    //CameraTransform cameraTransform(
    //  AnimatedTransform(worldFromCamera[0], graphicsState.transformStartTime,
    //    worldFromCamera[1], graphicsState.transformEndTime));
    //renderFromWorld = cameraTransform.RenderFromWorld();

    CameraSceneEntity camera;
    camera.name = name;
    camera.loc = loc;
    camera.outsideMedium = graphicsState.currentOutsideMedium;
    camera.dict.nOwnedParams = params.size();
    camera.dict.params = params.finalize();
    memcpy(&(camera.cameraFromWorld.m[0][0]), &(cameraFromWorld[0].m[0][0]), sizeof(Float) * 16);

    scene->camera = camera;
  }
  void BasicSceneBuilder::MakeNamedMedium(const std::string& origName, ParsedParameterVector params,
    FileLoc loc) {
    std::string name = origName;
    // Issue error if medium _name_ is multiply defined
    if (mediumNames.find(name) != mediumNames.end()) {
      ErrorExitDeferred(&loc, "Named medium \"%s\" redefined.", name);
      return;
    }
    mediumNames.insert(name);
    
    const class Transform renderFromObject = RenderFromObject(0);
    const class Transform objectFromRender = Inverse(renderFromObject);

    // Create _ParameterDictionary_ for medium and call _AddMedium()_
    MediumSceneEntity mediumEntity;
    memcpy(&(mediumEntity.renderFromObject.m), &(renderFromObject.m[0][0]), sizeof(Float) * 16);
    memcpy(&(mediumEntity.objectFromRender.m), &(objectFromRender.m[0][0]), sizeof(Float) * 16);
    
    mediumEntity.dict.nOwnedParams = params.size();
    mediumEntity.dict.params = params.finalize();
    std::vector<ParsedParameter*> medium_param_state =
      graphicsState.mediumAttributes.finalize();
    mediumEntity.dict.params.insert(mediumEntity.dict.params.end(),
      medium_param_state.begin(), medium_param_state.end());
    mediumEntity.loc = loc;
    mediumEntity.name = name;
    scene->AddMedium(mediumEntity);
  }
  void BasicSceneBuilder::MediumInterface(const std::string& insideName, const std::string& outsideName,
    FileLoc loc) {
    graphicsState.currentInsideMedium = insideName;
    graphicsState.currentOutsideMedium = outsideName;
  }
  void BasicSceneBuilder::WorldBegin(FileLoc loc) {

    // Reset graphics state for _WorldBegin_
    currentBlock = BlockState::WorldBlock;
    for (int i = 0; i < MaxTransforms; ++i)
      graphicsState.ctm[i] = tiny_pbrt_loader::Transform();
    graphicsState.activeTransformBits = AllTransformsBits;
    namedCoordinateSystems["world"] = graphicsState.ctm;

    // Pass pre-_WorldBegin_ entities to _scene_
    //scene->SetOptions(filter, film, camera, sampler, integrator, accelerator);
  }
  void BasicSceneBuilder::AttributeBegin(FileLoc loc) {
    pushedGraphicsStates.push_back(graphicsState);
    pushStack.push_back(std::make_pair('a', loc));
  }
  void BasicSceneBuilder::AttributeEnd(FileLoc loc) {
    // Issue error on unmatched _AttributeEnd_
    if (pushedGraphicsStates.empty()) {
      Error(&loc, "Unmatched AttributeEnd encountered. Ignoring it.");
      return;
    }

    // NOTE: must keep the following consistent with code in ObjectEnd
    graphicsState = std::move(pushedGraphicsStates.back());
    pushedGraphicsStates.pop_back();

    if (pushStack.back().first == 'o')
      ErrorExitDeferred(&loc,
        "Mismatched nesting: open ObjectBegin from %s at AttributeEnd",
        pushStack.back().second);
    else
      CHECK_EQ(pushStack.back().first, 'a');
    pushStack.pop_back();
  }

  void BasicSceneBuilder::Attribute(const std::string& target, ParsedParameterVector attrib,
    FileLoc loc) {
    ParsedParameterVector* currentAttributes = nullptr;
    if (target == "shape") {
      currentAttributes = &graphicsState.shapeAttributes;
    }
    else if (target == "light") {
      currentAttributes = &graphicsState.lightAttributes;
    }
    else if (target == "material") {
      currentAttributes = &graphicsState.materialAttributes;
    }
    else if (target == "medium") {
      currentAttributes = &graphicsState.mediumAttributes;
    }
    else if (target == "texture") {
      currentAttributes = &graphicsState.textureAttributes;
    }
    else {
      ErrorExitDeferred(
        &loc,
        "Unknown attribute target \"%s\". Must be \"shape\", \"light\", "
        "\"material\", \"medium\", or \"texture\".",
        target);
      return;
    }

    // Note that we hold on to the current color space and associate it
    // with the parameters...
    for (ParsedParameter* p : attrib) {
      //p->mayBeUnused = true;
      //p->colorSpace = graphicsState.colorSpace;
      currentAttributes->push_back(p);
    }
  }
  void BasicSceneBuilder::Texture(const std::string& name, const std::string& type,
    const std::string& texname, ParsedParameterVector params, FileLoc loc) {

  }
  void BasicSceneBuilder::Material(const std::string& name, ParsedParameterVector params, FileLoc loc) {
    SceneEntity materialEntity;
    materialEntity.dict.nOwnedParams = params.size();
    materialEntity.dict.params = params.finalize();
    std::vector<ParsedParameter*> material_param_state =
      graphicsState.materialAttributes.finalize();
    materialEntity.dict.params.insert(materialEntity.dict.params.end(),
      material_param_state.begin(), material_param_state.end());
    materialEntity.loc = loc;
    materialEntity.name = name;

    graphicsState.currentMaterialIndex = scene->AddMaterial(materialEntity);
    graphicsState.currentMaterialName.clear();
  }
  void BasicSceneBuilder::MakeNamedMaterial(const std::string& name, ParsedParameterVector params,
    FileLoc loc) {
    SceneEntity materialEntity;
    materialEntity.dict.nOwnedParams = params.size();
    materialEntity.dict.params = params.finalize();
    std::vector<ParsedParameter*> material_param_state =
      graphicsState.materialAttributes.finalize();
    materialEntity.dict.params.insert(materialEntity.dict.params.end(),
      material_param_state.begin(), material_param_state.end());
    if (namedMaterialNames.find(name) != namedMaterialNames.end()) {
      ErrorExitDeferred(&loc, "%s: named material redefined.", name);
      return;
    }
    namedMaterialNames.insert(name);
    scene->AddNamedMaterial(name, materialEntity);

  }
  void BasicSceneBuilder::NamedMaterial(const std::string& name, FileLoc loc) {
    graphicsState.currentMaterialName = name;
    graphicsState.currentMaterialIndex = -1;
  }
  void BasicSceneBuilder::LightSource(const std::string& name, ParsedParameterVector params, FileLoc loc) {
    //ParameterDictionary dict(std::move(params), graphicsState.lightAttributes,
    //  graphicsState.colorSpace);
    //scene->AddLight(LightSceneEntity(name, std::move(dict), loc, RenderFromObject(),
    //  graphicsState.currentOutsideMedium));
  }
  void BasicSceneBuilder::AreaLightSource(const std::string& name, ParsedParameterVector params,
    FileLoc loc) {
    graphicsState.areaLightName = name;
    graphicsState.areaLightParams = ParameterDictionary(std::move(params.finalize()));
    graphicsState.areaLightLoc = loc;
  }
  void BasicSceneBuilder::Shape(const std::string& name, ParsedParameterVector params, FileLoc loc) {
    ShapeSceneEntity shapeEntity;
    shapeEntity.dict.nOwnedParams = params.size();
    shapeEntity.dict.params = params.finalize();
    std::vector<ParsedParameter*> shape_param_state =
      graphicsState.shapeAttributes.finalize();
    shapeEntity.dict.params.insert(shapeEntity.dict.params.end(),
      shape_param_state.begin(), shape_param_state.end());
    shapeEntity.loc = loc;
    shapeEntity.name = name;

    int areaLightIndex = -1;
    if (!graphicsState.areaLightName.empty()) {
      //areaLightIndex = scene->AddAreaLight(SceneEntity(graphicsState.areaLightName,
      //  graphicsState.areaLightParams,
      //  graphicsState.areaLightLoc));
      //if (activeInstanceDefinition)
      //  Warning(&loc, "Area lights not supported with object instancing");
    }

    //if (CTMIsAnimated()) {
    //  AnimatedTransform renderFromShape = RenderFromObject();
    //  const class Transform* identity = transformCache.Lookup(pbrt::Transform());

    //  AnimatedShapeSceneEntity entity(
    //    { name, std::move(dict), loc, renderFromShape, identity,
    //     graphicsState.reverseOrientation, graphicsState.currentMaterialIndex,
    //     graphicsState.currentMaterialName, areaLightIndex,
    //     graphicsState.currentInsideMedium, graphicsState.currentOutsideMedium });

    //  if (activeInstanceDefinition)
    //    activeInstanceDefinition->entity.animatedShapes.push_back(std::move(entity));
    //  else
    //    scene->AddAnimatedShape(std::move(entity));
    //}
    //else 
    {
      const class Transform renderFromObject = RenderFromObject(0);
      const class Transform objectFromRender = Inverse(renderFromObject);

      memcpy(&(shapeEntity.renderFromObject.m), &(renderFromObject.m[0][0]), sizeof(Float) * 16);
      memcpy(&(shapeEntity.objectFromRender.m), &(objectFromRender.m[0][0]), sizeof(Float) * 16);

      shapeEntity.materialIndex = graphicsState.currentMaterialIndex;
      shapeEntity.outsideMedium = graphicsState.currentOutsideMedium;
      shapeEntity.insideMedium = graphicsState.currentInsideMedium;
      //(
      //  { name, std::move(dict), loc, renderFromObject, objectFromRender,
      //   graphicsState.reverseOrientation, graphicsState.currentMaterialIndex,
      //   graphicsState.currentMaterialName, areaLightIndex,
      //   graphicsState.currentInsideMedium, graphicsState.currentOutsideMedium });
      //if (activeInstanceDefinition)
      //  activeInstanceDefinition->entity.shapes.push_back(std::move(entity));
      //else
      shapes.push_back(std::move(shapeEntity));
    }
  }
  void BasicSceneBuilder::ReverseOrientation(FileLoc loc) {

  }
  void BasicSceneBuilder::ObjectBegin(const std::string& name, FileLoc loc) {

  }
  void BasicSceneBuilder::ObjectEnd(FileLoc loc) {

  }
  void BasicSceneBuilder::ObjectInstance(const std::string& name, FileLoc loc) {

  }

  void BasicSceneBuilder::EndOfFiles() {
    if (currentBlock != BlockState::WorldBlock)
      ErrorExitDeferred("End of files before \"WorldBegin\".");

    // Ensure there are no pushed graphics states
    while (!pushedGraphicsStates.empty()) {
      ErrorExitDeferred("Missing end to AttributeBegin");
      pushedGraphicsStates.pop_back();
    }

    if (errorExit)
      ErrorExit("Fatal errors during scene construction");

    if (!shapes.empty())
      scene->AddShapes(shapes);
    //if (!instanceUses.empty())
    //  scene->AddInstanceUses(instanceUses);
  }

  BasicSceneBuilder* BasicSceneBuilder::CopyForImport() {
    return nullptr;
  }

  void BasicSceneBuilder::MergeImported(BasicSceneBuilder*) {

  }

  std::string BasicSceneBuilder::ToString() const {
    return "";
  }


  void BasicScene::AddNamedMaterial(std::string name, SceneEntity material) {
    namedMaterials.push_back(std::make_pair(std::move(name), std::move(material)));
  }

  int BasicScene::AddMaterial(SceneEntity material) {
    materials.push_back(std::move(material));
    return int(materials.size() - 1);
  }

  //void BasicScene::AddLight(LightSceneEntity light) {
  //  Medium lightMedium = GetMedium(light.medium, &light.loc);
  //  std::lock_guard<std::mutex> lock(lightMutex);

  //  if (light.renderFromObject.IsAnimated())
  //    Warning(&light.loc,
  //      "Animated lights aren't supported. Using the start transform.");

  //  auto create = [this, light, lightMedium]() {
  //    return Light::Create(light.name, light.parameters,
  //      light.renderFromObject.startTransform,
  //      GetCamera().GetCameraTransform(), lightMedium, &light.loc,
  //      threadAllocators.Get());
  //  };
  //  lightJobs.push_back(RunAsync(create));
  //}

  int BasicScene::AddAreaLight(SceneEntity light) {
    areaLights.push_back(std::move(light));
    return areaLights.size() - 1;
  }

  void BasicScene::AddShapes(pstd::span<ShapeSceneEntity> s) {
    std::move(std::begin(s), std::end(s), std::back_inserter(shapes));
  }

  //void BasicScene::AddAnimatedShape(AnimatedShapeSceneEntity shape) {
  //  //animatedShapes.push_back(std::move(shape));
  //}
  //
  void BasicScene::AddMedium(MediumSceneEntity medium) {
    mediums.push_back(std::move(medium));
  }

  void ParameterDictionary::FreeParameters() {
    for (int i = 0; i < nOwnedParams; ++i)
      delete params[i];
    params.clear();
  }

  std::unique_ptr<BasicScene> load_scene_from_string(std::string str, std::string dir_path) {
    std::unique_ptr<BasicScene> scene = std::make_unique<BasicScene>();
    path_of_the_main_file = dir_path;
    BasicSceneBuilder target(scene.get());
    ParseString(&target, str);
    return std::move(scene);
  }

  template <ParameterType PT>
  typename ParameterTypeTraits<PT>::ReturnType ParameterDictionary::lookupSingle(
    const std::string& name,
    typename ParameterTypeTraits<PT>::ReturnType defaultValue) const {
    // Search _params_ for parameter _name_
    using traits = ParameterTypeTraits<PT>;
    for (const ParsedParameter* p : params) {
      if (p->name != name || p->type != traits::typeName)
        continue;
      // Extract parameter values from _p_
      const auto& values = traits::GetValues(*p);

      // Issue error if an incorrect number of parameter values were provided
      if (values.empty())
        ErrorExit(&p->loc, "No values provided for parameter \"%s\".", name);
      if (values.size() != traits::nPerItem)
        ErrorExit(&p->loc, "Expected %d values for parameter \"%s\".",
          traits::nPerItem, name);

      // Return parameter values as _ReturnType_
      p->lookedUp = true;
      return traits::Convert(values.data(), &p->loc);
    }

    return defaultValue;
  }

  Float ParameterDictionary::GetOneFloat(const std::string& name, Float def) const {
    return lookupSingle<ParameterType::Float>(name, def);
  }

  int ParameterDictionary::GetOneInt(const std::string& name, int def) const {
    return lookupSingle<ParameterType::Integer>(name, def);
  }

  bool ParameterDictionary::GetOneBool(const std::string& name, bool def) const {
    return lookupSingle<ParameterType::Boolean>(name, def);
  }

  Point2f ParameterDictionary::GetOnePoint2f(const std::string& name, Point2f def) const {
    return lookupSingle<ParameterType::Point2f>(name, def);
  }

  Vector2f ParameterDictionary::GetOneVector2f(const std::string& name,
    Vector2f def) const {
    return lookupSingle<ParameterType::Vector2f>(name, def);
  }

  Vector3f ParameterDictionary::GetOneVector3f(const std::string& name,
    Vector3f def) const {
    return lookupSingle<ParameterType::Vector3f>(name, def);
  }

  Vector3f ParameterDictionary::GetOneRGB3f(const std::string& name,
    Vector3f def) const {
    return lookupSingle<ParameterType::RGB>(name, def);
  }

  //Vector3f ParameterDictionary::GetOnePoint3f(const std::string& name,
  //  Point3f def) const {
  //  return lookupSingle<ParameterType::Point3f>(name, def);
  //}

  Normal3f ParameterDictionary::GetOneNormal3f(const std::string& name,
    Normal3f def) const {
    return lookupSingle<ParameterType::Normal3f>(name, def);
  }

  std::string ParameterDictionary::GetOneString(const std::string& name,
    const std::string& def) const {
    return lookupSingle<ParameterType::String>(name, def);
  }

  template <typename ReturnType, typename ValuesType, typename C>
  static std::vector<ReturnType> returnArray(const ValuesType& values,
    const ParsedParameter& param, int nPerItem,
    C convert) {
    if (values.empty())
      ErrorExit(&param.loc, "No values provided for \"%s\".", param.name);
    if (values.size() % nPerItem)
      ErrorExit(&param.loc, "Number of values provided for \"%s\" not a multiple of %d",
        param.name, nPerItem);

    param.lookedUp = true;
    size_t n = values.size() / nPerItem;
    std::vector<ReturnType> v(n);
    for (size_t i = 0; i < n; ++i)
      v[i] = convert(&values[nPerItem * i], &param.loc);
    return v;
  }

  template <typename ReturnType, typename G, typename C>
  std::vector<ReturnType> ParameterDictionary::lookupArray(const std::string& name,
    ParameterType type,
    const char* typeName,
    int nPerItem, G getValues,
    C convert) const {
    for (const ParsedParameter* p : params)
      if (p->name == name && p->type == typeName)
        return returnArray<ReturnType>(getValues(*p), *p, nPerItem, convert);
    return {};
  }

  template <ParameterType PT>
  std::vector<typename ParameterTypeTraits<PT>::ReturnType>
    ParameterDictionary::lookupArray(const std::string& name) const {
    using traits = ParameterTypeTraits<PT>;
    return lookupArray<typename traits::ReturnType>(
      name, PT, traits::typeName, traits::nPerItem, traits::GetValues, traits::Convert);
  }

  std::vector<Float> ParameterDictionary::GetFloatArray(const std::string& name) const {
    return lookupArray<ParameterType::Float>(name);
  }

  std::vector<int> ParameterDictionary::GetIntArray(const std::string& name) const {
    return lookupArray<ParameterType::Integer>(name);
  }

  std::vector<uint8_t> ParameterDictionary::GetBoolArray(const std::string& name) const {
    return lookupArray<ParameterType::Boolean>(name);
  }

  std::vector<Point2f> ParameterDictionary::GetPoint2fArray(const std::string& name) const {
    return lookupArray<ParameterType::Point2f>(name);
  }

  std::vector<Vector2f> ParameterDictionary::GetVector2fArray(
    const std::string& name) const {
    return lookupArray<ParameterType::Vector2f>(name);
  }

  std::vector<Point3f> ParameterDictionary::GetPoint3fArray(const std::string& name) const {
    return lookupArray<ParameterType::Point3f>(name);
  }

  std::vector<Vector3f> ParameterDictionary::GetVector3fArray(
    const std::string& name) const {
    return lookupArray<ParameterType::Vector3f>(name);
  }

  std::vector<Normal3f> ParameterDictionary::GetNormal3fArray(
    const std::string& name) const {
    return lookupArray<ParameterType::Normal3f>(name);
  }

  std::vector<Float> const& ParameterDictionary::GetAllFloats(const std::string& name) const {
    for (const ParsedParameter* p : params)
      if (p->name == name)
        return p->floats;
    return {};
  }

  static std::map<std::string, Spectrum> cachedSpectra;
}