#ifndef TABLE_HPP
#define TABLE_HPP

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// TODO:
//  Use aligned vector for table data
//  Use memory arena for the storage vectors

namespace aks {
using size_t = std::size_t;
namespace table_detail {}

template <typename... Ts> struct table;

template <typename... Ts> struct const_table_row {
  using data_type = std::tuple<std::reference_wrapper<Ts const>...>;
  data_type items;
};

template <std::size_t N, typename... Ts>
decltype(auto) get(const_table_row<Ts...> const &t) {
  return std::get<N>(t.items).get();
}

template <typename... Ts> struct table_row {
  using data_type = std::tuple<std::reference_wrapper<Ts>...>;
  data_type items;

  operator const_table_row<Ts...>() { return {items}; }
};

template <std::size_t N, typename... Ts>
decltype(auto) get(table_row<Ts...> const &t) {
  return std::get<N>(t.items).get();
}

template <typename... Ts> struct table_row_copy {
  using data_type = std::tuple<Ts...>;
  std::tuple<Ts...> items;
};

template <std::size_t N, typename... Ts>
decltype(auto) get(table_row_copy<Ts...> const &t) {
  return std::get<N>(t.items);
}

template <typename... Ts> struct row_id {
  table<Ts...> const *tbl = nullptr;
  size_t id = std::numeric_limits<size_t>::max();
};

template <typename... Ts> struct const_table_iterator {
  table<Ts...> const *tbl = nullptr;
  size_t id = std::numeric_limits<size_t>::max();

  const_table_row<Ts...> operator*() const { return (*tbl)[id]; }

  const_table_iterator &operator++() {
    ++id;
    return *this;
  }

  const_table_iterator operator++(int) {
    const_table_iterator tmp{*this};
    ++(*this);
    return tmp;
  }

  const_table_iterator &operator--() {
    --id;
    return *this;
  }

  const_table_iterator operator--(int) {
    const_table_iterator tmp{*this};
    --(*this);
    return tmp;
  }

  bool operator==(const_table_iterator const &other) const {
    return tbl == other.tbl && id == other.id;
  }

  bool operator!=(const_table_iterator const &other) const {
    return !((*this) == other);
  }
};

template <typename... Ts> struct table_iterator {
  table<Ts...> *tbl = nullptr;
  size_t id = std::numeric_limits<size_t>::max();

  table_row<Ts...> operator*() { return (*tbl)[id]; }

  table_iterator &operator++() {
    ++id;
    return *this;
  }

  table_iterator operator++(int) {
    table_iterator tmp{*this};
    ++(*this);
    return tmp;
  }

  table_iterator &operator--() {
    --id;
    return *this;
  }

  table_iterator operator--(int) {
    table_iterator tmp{*this};
    --(*this);
    return tmp;
  }

  bool operator==(table_iterator const &other) const {
    return tbl == other.tbl && id == other.id;
  }

  bool operator!=(table_iterator const &other) const {
    return !((*this) == other);
  }

  operator const_table_iterator<Ts...>() { return {tbl, id}; }
};

template <typename... Ts> struct table {
  template <typename T> using vec_t = std::vector<T>;
  using data_type = std::tuple<vec_t<Ts>...>;
  constexpr static const size_t column_count = sizeof...(Ts);
  constexpr static const size_t _sentinel = std::numeric_limits<size_t>::max();
  template <size_t N_>
  using column_type =
      typename std::tuple_element<N_, data_type>::type::value_type;

  using row_id_type = row_id<Ts...>;
  using const_table_row_type = const_table_row<Ts...>;
  using table_row_type = table_row<Ts...>;
  using table_iterator_type = table_iterator<Ts...>;
  using const_table_iterator_type = const_table_iterator<Ts...>;

  table const &as_const() const { return (*this); }

  table(table &&) = default;
  table(table const &) = default;
  table() = default;
  table &operator=(table const &) = default;
  table &operator=(table &&) = default;

private:
  data_type data{};
  vec_t<size_t> _index{};
  vec_t<size_t> _actual_index{};

  template <typename... Us> friend struct table;

  table(data_type dt, vec_t<size_t> index, vec_t<size_t> actual_index)
      : data(std::move(dt)), _index(std::move(index)),
        _actual_index(std::move(actual_index)) {}

  size_t push_back_helper() {
    size_t const sz = size();
    size_t const asz = _actual_index.size();
    _index.push_back(asz);
    _actual_index.push_back(sz);
    return asz;
  }

public:
  template <size_t I> auto &get() { return std::get<I>(data); }
  template <size_t I> auto const &get() const { return std::get<I>(data); }

  void reserve(size_t N) {
    auto reserve_impl = [=](vec_t<Ts> &...args) { (args.reserve(N), ...); };
    std::apply(reserve_impl, data);
    _index.reserve(N);
    _actual_index.reserve(N);
  }

  row_id<Ts...> push_back(Ts const &...ts) {
    auto push_back_impl = [&](vec_t<Ts> &...args) {
      (args.push_back(ts), ...);
    };
    size_t const asz = push_back_helper();
    std::apply(push_back_impl, data);
    return {this, asz};
  }

  row_id<Ts...> emplace_back(Ts &&...ts) {
    auto emplace_back_impl = [&](vec_t<Ts> &...args) mutable {
      (args.emplace_back(std::move(ts)), ...);
    };
    size_t const asz = push_back_helper();
    std::apply(emplace_back_impl, data);
    return {this, asz};
  }

  table_row_copy<Ts...> remove(row_id_type r) {
    size_t const i = r.id;
    size_t const idx = _actual_index[i];
    _index[idx] = _index.back();
    _actual_index[_index[idx]] = idx;
    _actual_index[i] = _sentinel;
    _index.pop_back();

    auto set_idx_to_back_aka_compact = [&](auto &arg) {
      arg[idx] = std::move(arg.back());
      arg.pop_back();
    };

    auto pop_back_impl = [&](vec_t<Ts> &...args) {
      using data_type = typename table_row_copy<Ts...>::data_type;
      auto ret = data_type(std::move(args.at(idx))...);
      (set_idx_to_back_aka_compact(args), ...);
      return ret;
    };
    return table_row_copy<Ts...>{std::apply(pop_back_impl, data)};
  }

  table_row<Ts...> operator[](size_t n) {
    size_t const idx = n;
    auto at_impl = [&](vec_t<Ts> &...args) {
      using data_type = typename table_row<Ts...>::data_type;
      return data_type(args.at(idx)...);
    };
    return table_row<Ts...>{std::apply(at_impl, data)};
  }

  const_table_row<Ts...> operator[](size_t n) const {
    size_t const idx = n;
    auto at_impl = [&](vec_t<Ts> const &...args) {
      using data_type = typename const_table_row<Ts...>::data_type;
      return data_type(args.at(idx)...);
    };
    return const_table_row<Ts...>{std::apply(at_impl, data)};
  }

  table_row<Ts...> operator[](row_id_type r) {
    size_t const idx = _actual_index[r.id];
    auto at_impl = [&](vec_t<Ts> &...args) {
      using data_type = typename table_row<Ts...>::data_type;
      return data_type(args.at(idx)...);
    };
    return table_row<Ts...>{std::apply(at_impl, data)};
  }

  const_table_row<Ts...> operator[](row_id_type r) const {
    size_t const idx = _actual_index[r.id];
    auto at_impl = [&](vec_t<Ts> const &...args) {
      using data_type = typename const_table_row<Ts...>::data_type;
      return data_type(args.at(idx)...);
    };
    return const_table_row<Ts...>{std::apply(at_impl, data)};
  }

  std::optional<const_table_row<Ts...>> at(row_id<Ts...> const &r) const {
    if (r.tbl != this) {
      return std::nullopt;
    }
    size_t const idx = _actual_index[r.id];
    auto at_impl = [&](vec_t<Ts> const &...args) {
      using data_type = typename const_table_row<Ts...>::data_type;
      return data_type(args.at(idx)...);
    };
    return const_table_row<Ts...>{std::apply(at_impl, data)};
  }

  std::optional<table_row<Ts...>> at(row_id<Ts...> const &r) {
    if (r.tbl != this) {
      return std::nullopt;
    }
    size_t const idx = _actual_index[r.id];
    auto at_impl = [&](vec_t<Ts> &...args) {
      using data_type = typename table_row<Ts...>::data_type;
      return data_type(args.at(idx)...);
    };
    return table_row<Ts...>{std::apply(at_impl, data)};
  }

  std::optional<const_table_row<Ts...>> at_const(row_id<Ts...> const &r) const {
    return at(r);
  }

  size_t size() const { return std::get<0>(data).size(); }

  template <size_t... Sz> auto sub_table() const {
    using ret_t =
        table<typename std::tuple_element<Sz, data_type>::type::value_type...>;
    return ret_t{typename ret_t::data_type(std::get<Sz>(data)...), _index,
                 _actual_index};
  }

  const_table_iterator<Ts...> begin() const { return {this, 0}; }

  const_table_iterator<Ts...> end() const { return {this, size()}; }

  table_iterator<Ts...> begin() { return {this, 0}; }

  table_iterator<Ts...> end() { return {this, size()}; }

  const_table_iterator<Ts...> cbegin() const { return {this, 0}; }

  const_table_iterator<Ts...> cend() const { return {this, size()}; }

  template <typename F> std::optional<table_row_type> find(F f) {
    auto f_impl = [&](Ts const &...args) { return f(args...); };

    for (auto row : (*this)) {
      if (std::apply(f_impl, row.items)) {
        return row;
      }
    }

    return std::nullopt;
  }

  template <size_t I, typename F>
  std::optional<const_table_row_type> find_if(F f) const {
    auto const &xs = std::get<I>(data);

    auto it = std::find_if(xs.begin(), xs.end(), f);

    if (it != xs.end()) {
      auto idx = std::distance(xs.begin(), it);
      return (*this)[idx];
    }

    return std::nullopt;
  }

  template <typename F> std::optional<const_table_row_type> find(F f) const {
    auto f_impl = [&](Ts const &...args) { return f(args...); };

    for (auto const &row : (*this)) {
      if (std::apply(f_impl, row.items)) {
        return row;
      }
    }

    return std::nullopt;
  }

  template <typename F> table<Ts...> filtered(F f) const {
    table<Ts...> new_table;

    auto f_impl = [&](Ts const &...args) {
      if (f(args...)) {
        new_table.push_back(args...);
      }
    };

    for (auto const &row : (*this)) {
      std::apply(f_impl, row.items);
    }

    return new_table;
  }

  template <typename F> vec_t<row_id_type> matching_row_ids(F f) const {
    vec_t<row_id_type> new_result;
    auto f_impl = [&](Ts const &...args) -> bool { return f(args...); };

    for (size_t i = 0; i < size(); ++i) {
      if (std::apply(f_impl, (*this)[i].items)) {
        row_id_type new_id = {this, _index[i]};
        new_result.push_back(new_id);
      }
    }

    return new_result;
  }

  template <typename F> vec_t<const_table_row_type> filter(F f) const {
    vec_t<const_table_row_type> new_result;
    auto f_impl = [&](Ts const &...args) -> bool { return f(args...); };

    for (auto row : (*this)) {
      if (std::apply(f_impl, row.items)) {
        new_result.emplace_back(std::move(row));
      }
    }

    return new_result;
  }

  template <typename F> auto map(F f) const {
    using f_return_type = decltype(f(std::declval<Ts>()...));
    auto map_impl = [&](auto... f_rets) {
      table<decltype(f_rets)...> new_table;
      auto f_impl = [&](Ts const &...row_args) {
        auto pb_impl = [&](auto &&...new_data_args) {
          new_table.push_back(new_data_args...);
        };
        std::apply(pb_impl, f(row_args...));
      };
      for (auto const &row : (*this)) {
        std::apply(f_impl, row.items);
      }
      return new_table;
    };
    return std::apply(map_impl, f_return_type{});
  }

  void vappend_inplace(table const &other) {
    // other can be the same as "this", in which case the insert can cause a
    // problem on gcc and clang debugs, so we should not do that!

    // this will mess up all the stable ids... can't think of a way to make that
    // work.
    // TODO:
    // put a clean index and actual index in there... so it is as-if they were
    // pushed into a clean table (or into this table on top, to actual_index and
    // index will change accordingly)

    auto vappend_impl = [&](std::vector<Ts> &...vst) {
      auto va_internal = [&](std::vector<Ts> const &...vso) {
        (vst.insert(vst.end(), vso.begin(), vso.end()), ...);
      };
      std::apply(va_internal, other.data);
    };

    std::apply(vappend_impl, data);
  }

  template <typename... Us>
  auto
  happend(table<Us...> const &other) -> std::optional<table<Ts..., Us...>> {
    // this will mess up all the stable ids... can't think of a way to make that
    // work.
    // TODO:
    // put a clean index and actual index in there... so it is as-if they were
    // pushed into a clean table
    // or keep the current stable ids (as is currently the case)

    if (this->size() != other.size()) {
      return std::nullopt;
    }

    auto happend_impl = [&](std::vector<Ts> const &...vst) {
      auto ha_internal = [&](std::vector<Us> const &...vso) {
        return table<Ts..., Us...>{
            typename table<Ts..., Us...>::data_type(vst..., vso...), _index,
            _actual_index};
      };
      return std::apply(ha_internal, other.data);
    };

    return std::apply(happend_impl, data);
  }

  auto vappend(table const &other) const {
    table new_table = (*this);
    new_table.vappend_inplace(other);
    return new_table;
  }
};

template <typename... Ts> auto begin(table<Ts...> &t) { return t.begin(); }
template <typename... Ts> auto begin(table<Ts...> const &t) {
  return t.begin();
}
template <typename... Ts> auto cbegin(table<Ts...> &t) { return t.cbegin(); }
template <typename... Ts> auto cbegin(table<Ts...> const &t) {
  return t.cbegin();
}
template <typename... Ts> auto end(table<Ts...> &t) { return t.end(); }
template <typename... Ts> auto end(table<Ts...> const &t) { return t.end(); }
template <typename... Ts> auto cend(table<Ts...> &t) { return t.cend(); }
template <typename... Ts> auto cend(table<Ts...> const &t) { return t.cend(); }
} // namespace aks

namespace std {
template <typename... Ts>
struct tuple_size<aks::table_row<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)> {};

template <typename... Ts>
struct tuple_size<aks::const_table_row<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)> {};

template <typename... Ts>
struct tuple_size<aks::table_row_copy<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)> {};

template <size_t Index, typename... Ts>
struct tuple_element<Index, aks::table_row<Ts...>> {
  using data_type = typename aks::table_row<Ts...>::data_type;
  using type = decltype(std::get<Index>(declval<data_type>()).get());
};

template <size_t Index, typename... Ts>
struct tuple_element<Index, const aks::table_row<Ts...>> {
  using table_type = aks::table_row<Ts...> const;
  using data_type = typename table_type::data_type;
  using type =
      decltype(std::cref(std::get<Index>(declval<data_type>()).get()).get());
};

template <size_t Index, typename... Ts>
struct tuple_element<Index, aks::const_table_row<Ts...>> {
  using data_type = typename aks::const_table_row<Ts...>::data_type;
  using type = decltype(std::get<Index>(declval<data_type>()).get());
};

template <size_t Index, typename... Ts>
struct tuple_element<Index, aks::table_row_copy<Ts...>>
    : tuple_element<Index, typename aks::table_row<Ts...>::data_type> {};

} // namespace std

#endif // TABLE_HPP

#ifndef TABLE_IO_HPP
#define TABLE_IO_HPP

#include <iostream>

namespace aks {

template <typename... Ts>
std::ostream &operator<<(std::ostream &o, table_row<Ts...> const &row) {
  auto cout_impl = [&o](Ts const &...tupleArgs) {
    o << '[';
    std::size_t n{0};
    ((o << tupleArgs << (++n != sizeof...(Ts) ? ", " : "")), ...);
    o << ']';
  };

  std::apply(cout_impl, row.items);
  return o;
}

template <typename... Ts>
std::ostream &operator<<(std::ostream &o, const_table_row<Ts...> const &row) {
  auto cout_impl = [&o](Ts const &...tupleArgs) {
    o << '[';
    std::size_t n{0};
    ((o << tupleArgs << (++n != sizeof...(Ts) ? ", " : "")), ...);
    o << ']';
  };

  std::apply(cout_impl, row.items);
  return o;
}

template <typename... Ts>
std::ostream &operator<<(std::ostream &o, table<Ts...> const &tbl) {
  o << "----- TABLE SIZE = " << tbl.size() << " -----\n";
  for (auto const &row : tbl) {
    std::cout << row << "\n";
  }
  o << "- - - - - - - - - - - - - - -\n";
  return o;
}

} // namespace aks

#endif // TABLE_IO_HPP
