// SPDX-License-Identifier: MIT
//
// Measure wall-clock selection time of:
//   origami::rank_configs              (analytical scorer)
//   origami::ml_recommender::rank_configs  (ML two-tower)
//
// Inputs come from a TSV that bench_selection_time.py produces by
// parsing a Tensile Origami UserArgs.yaml -- so the candidate set
// is the actual library hipBLASLt would dispatch from.
//
// Output is one line per GEMM shape:
//   M N K B  | n_kernels |  origami_us  ml_us  speedup

#include "origami/hardware.hpp"
#include "origami/origami.hpp"
#include "origami/types.hpp"

#include "tilewright/hipblaslt/adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Shape {
  std::size_t m, n, k, b;
};

struct ConfigRow {
  int mt_m, mt_n, mt_k;
  int mi_m, mi_n, mi_k;
  int occ;
  int cha, chb;
  int grvwa, grvwb, gwvwd;
  int chc, chd, che;
  int pgr, plr, lrvw;
  int lpa, lpb, lbspa, lbspb;
  int lsu;
};

struct Inputs {
  int a_dtype, b_dtype, c_dtype, d_dtype, mi_dtype;
  int trans_a, trans_b;
  std::vector<Shape> shapes;
  std::vector<ConfigRow> configs;
};

bool _next_data_line(std::ifstream& in, std::string& out) {
  while (std::getline(in, out)) {
    auto a = out.find_first_not_of(" \t\r\n");
    if (a == std::string::npos || out[a] == '#') continue;
    return true;
  }
  return false;
}

Inputs load_inputs(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  Inputs ix;
  std::string line, tag;

  if (!_next_data_line(in, line)) std::exit(1);
  {
    std::istringstream s(line);
    s >> tag >> ix.a_dtype >> ix.b_dtype >> ix.c_dtype >> ix.d_dtype >> ix.mi_dtype >> ix.trans_a >>
        ix.trans_b;
    if (tag != "PROBLEM_HEADER") {
      std::fprintf(stderr, "expected PROBLEM_HEADER, got %s\n", tag.c_str());
      std::exit(1);
    }
  }

  std::size_t np = 0, nc = 0;
  if (!_next_data_line(in, line)) std::exit(1);
  {
    std::istringstream s(line);
    s >> tag >> np;
  }
  for (std::size_t i = 0; i < np; ++i) {
    if (!_next_data_line(in, line)) std::exit(1);
    std::istringstream s(line);
    Shape sh{};
    s >> sh.m >> sh.n >> sh.k >> sh.b;
    if (!s) std::exit(1);
    ix.shapes.push_back(sh);
  }

  if (!_next_data_line(in, line)) std::exit(1);
  {
    std::istringstream s(line);
    s >> tag >> nc;
  }
  for (std::size_t i = 0; i < nc; ++i) {
    if (!_next_data_line(in, line)) std::exit(1);
    std::istringstream s(line);
    ConfigRow c{};
    s >> c.mt_m >> c.mt_n >> c.mt_k >> c.mi_m >> c.mi_n >> c.mi_k >> c.occ >> c.cha >> c.chb >>
        c.grvwa >> c.grvwb >> c.gwvwd >> c.chc >> c.chd >> c.che >> c.pgr >> c.plr >> c.lrvw >>
        c.lpa >> c.lpb >> c.lbspa >> c.lbspb >> c.lsu;
    if (!s) std::exit(1);
    ix.configs.push_back(c);
  }
  return ix;
}

void fill_config(origami::config_t& c,
                 const ConfigRow& r,
                 std::size_t idx,
                 origami::prediction_modes_t mode) {
  c                = origami::config_t{};
  c.mt             = origami::dim3_t{(std::size_t)r.mt_m, (std::size_t)r.mt_n, (std::size_t)r.mt_k};
  c.mi             = origami::dim3_t{(std::size_t)r.mi_m, (std::size_t)r.mi_n, (std::size_t)r.mi_k};
  c.occupancy      = std::max(1, r.occ);
  c.cache_hints_a  = r.cha;
  c.cache_hints_b  = r.chb;
  c.grvw_a         = (std::size_t)std::max(1, r.grvwa);
  c.grvw_b         = (std::size_t)std::max(1, r.grvwb);
  c.gwvw_d         = (std::size_t)std::max(1, r.gwvwd);
  c.vector_width_a = std::max(1, r.lrvw);
  c.vector_width_b = std::max(1, r.lrvw);
  c.workgroup_mapping     = 0;
  c.prediction_mode       = mode;
  c.target                = origami::target_t::tensilelite;
  c.grid_selection        = origami::grid_selection_t::k_split_aware;
  c.index                 = idx;
  auto& tp                = c.tensile();
  tp.depth_u              = (std::size_t)r.mt_k;
  tp.global_split_u       = 1;
  tp.local_split_u        = std::max(1, r.lsu);
  tp.prefetch_global_read = std::max(1, r.pgr);
}

template <typename F>
double median_us(std::size_t warmup, std::size_t iters, F&& f) {
  for (std::size_t i = 0; i < warmup; ++i) f();
  std::vector<double> us;
  us.reserve(iters);
  for (std::size_t i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto r  = f();
    auto t1 = std::chrono::steady_clock::now();
    asm volatile("" : : "g"(r.size()) : "memory");
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s --weights PATH --inputs TSV "
               "[--iters 200] [--warmup 50] [--arch gfx950]\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path, inputs_path, arch = "gfx950";
  std::size_t iters = 200, warmup = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need     = [&](int k) {
      if (i + k >= argc) {
        usage(argv[0]);
        std::exit(2);
      }
    };
    if (a == "--weights") {
      need(1);
      weights_path = argv[++i];
    } else if (a == "--inputs") {
      need(1);
      inputs_path = argv[++i];
    } else if (a == "--iters") {
      need(1);
      iters = std::stoul(argv[++i]);
    } else if (a == "--warmup") {
      need(1);
      warmup = std::stoul(argv[++i]);
    } else if (a == "--arch") {
      need(1);
      arch = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      usage(argv[0]);
      return 2;
    }
  }
  if (weights_path.empty() || inputs_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  if (!tilewright::load_weights(weights_path)) {
    std::fprintf(stderr, "FATAL: failed to load weights from %s\n", weights_path.c_str());
    return 1;
  }
  auto ix = load_inputs(inputs_path);

  auto arch_e = origami::hardware_t::arch_name_to_enum(arch);
  if (arch_e == origami::hardware_t::architecture_t::Count) {
    std::fprintf(stderr, "unknown arch: %s\n", arch.c_str());
    return 1;
  }
  // (arch, N_CU, lds_capacity, rf_capacity, L2_capacity, compute_clock_khz).
  // rf_capacity (register-file bytes/CU) is part of the origami signature;
  // gfx950 (CDNA4) has 512 KiB VGPR/CU. These are representative
  // values -- this is a CPU timing microbench, so the exact capacities only
  // affect the feasibility filter, not the rank_configs() wall-clock.
  auto hw =
      origami::hardware_t::get_hardware_for_arch(arch_e, 256, 65536, 524288, 4194304, 2100000);

  const auto ta       = (ix.trans_a == 0) ? origami::transpose_t::T : origami::transpose_t::N;
  const auto tb       = (ix.trans_b == 0) ? origami::transpose_t::T : origami::transpose_t::N;
  const std::size_t N = ix.configs.size();

  // Build the candidate vectors ONCE -- per-shape, only `problem_t`
  // changes, the kernel list is the whole library.
  std::vector<origami::config_t> cfgs_est(N), cfgs_ml(N);
  // Pre-built tilewright configs (base params + per-config ML features), index-
  // aligned with cfgs_ml -- mirrors how the deployed PredictionLibrary builds
  // its tilewright_config_list, so the bench picks match the runtime.
  std::vector<tilewright::Config> mcfgs(N);
  for (std::size_t i = 0; i < N; ++i) {
    fill_config(cfgs_est[i], ix.configs[i], i, origami::prediction_modes_t::estimation);
    fill_config(cfgs_ml[i], ix.configs[i], i, origami::prediction_modes_t::estimation);
    const ConfigRow& r     = ix.configs[i];
    tilewright::Config& mc = mcfgs[i];
    mc.mt        = tilewright::Dim3{(std::size_t)r.mt_m, (std::size_t)r.mt_n, (std::size_t)r.mt_k};
    mc.mi        = tilewright::Dim3{(std::size_t)r.mi_m, (std::size_t)r.mi_n, (std::size_t)r.mi_k};
    mc.occupancy = std::max(1, r.occ);
    mc.cache_hints_a = r.cha;
    mc.cache_hints_b = r.chb;
    mc.grvw_a        = (std::size_t)std::max(1, r.grvwa);
    mc.grvw_b        = (std::size_t)std::max(1, r.grvwb);
    mc.gwvw_d        = (std::size_t)std::max(1, r.gwvwd);
    mc.index         = i;
    // The config_ml_t ML fields (cache_hints_c/d/e, prefetch_*, LDS-RVW/pads,
    // local_split_u) were removed from tilewright::Config (dead feature group
    // dropped); nothing to set here anymore.
  }

  std::fprintf(stderr,
               "selection time over %zu shapes, %zu kernels per call, "
               "%zu iters per shape (median, after %zu warmup), arch=%s\n",
               ix.shapes.size(),
               N,
               iters,
               warmup,
               arch.c_str());

  std::vector<double> origami_us, ml_us;
  origami_us.reserve(ix.shapes.size());
  ml_us.reserve(ix.shapes.size());
  std::size_t origami_skipped = 0, ml_skipped = 0;

  for (std::size_t si = 0; si < ix.shapes.size(); ++si) {
    const auto& sh = ix.shapes[si];
    origami::problem_t p{};
    p.size        = origami::dim3_t{sh.m, sh.n, sh.k};
    p.batch       = sh.b;
    p.a_transpose = ta;
    p.b_transpose = tb;
    p.a_dtype     = (origami::data_type_t)ix.a_dtype;
    p.b_dtype     = (origami::data_type_t)ix.b_dtype;
    p.c_dtype     = (origami::data_type_t)ix.c_dtype;
    p.d_dtype     = (origami::data_type_t)ix.d_dtype;
    p.mi_dtype    = (origami::data_type_t)ix.mi_dtype;

    try {
      origami_us.push_back(
          median_us(warmup, iters, [&]() { return origami::rank_configs(p, hw, cfgs_est); }));
    } catch (const std::exception&) { ++origami_skipped; }
    try {
      ml_us.push_back(median_us(warmup, iters, [&]() {
        return tilewright::hipblaslt::rank_configs(p, hw, mcfgs, cfgs_ml);
      }));
    } catch (const std::exception&) { ++ml_skipped; }

    // tiny progress dot to stderr every 50 shapes (kept off stdout)
    if ((si + 1) % 50 == 0) {
      std::fprintf(stderr, "  ... %zu/%zu shapes\n", si + 1, ix.shapes.size());
    }
  }

  auto summarize = [](std::vector<double> v) {
    struct R {
      double mn, mx, med, mean, p90, p99;
      std::size_t n;
    };
    R r{0, 0, 0, 0, 0, 0, v.size()};
    if (v.empty()) return r;
    std::sort(v.begin(), v.end());
    auto pct = [&](double q) {
      if (v.size() == 1) return v[0];
      double idx     = q * static_cast<double>(v.size() - 1);
      std::size_t lo = static_cast<std::size_t>(idx);
      double frac    = idx - static_cast<double>(lo);
      std::size_t hi = (lo + 1 < v.size()) ? lo + 1 : lo;
      return v[lo] + (v[hi] - v[lo]) * frac;
    };
    r.mn       = v.front();
    r.mx       = v.back();
    r.med      = pct(0.50);
    r.p90      = pct(0.90);
    r.p99      = pct(0.99);
    double sum = 0;
    for (double x : v) sum += x;
    r.mean = sum / static_cast<double>(v.size());
    return r;
  };
  auto o = summarize(origami_us);
  auto m = summarize(ml_us);

  std::printf("# shapes=%zu  kernels_per_call=%zu  iters=%zu (median)  arch=%s\n",
              ix.shapes.size(),
              N,
              iters,
              arch.c_str());
  std::printf("# all numbers are wall-clock microseconds per rank_configs() call\n");
  std::printf("%-10s  %-7s  %-10s %-10s %-10s %-10s %-10s %-10s\n",
              "path",
              "n",
              "min",
              "median",
              "mean",
              "p90",
              "p99",
              "max");
  std::printf("%-10s  %-7zu  %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n",
              "origami",
              o.n,
              o.mn,
              o.med,
              o.mean,
              o.p90,
              o.p99,
              o.mx);
  std::printf("%-10s  %-7zu  %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n",
              "ml_rec",
              m.n,
              m.mn,
              m.med,
              m.mean,
              m.p90,
              m.p99,
              m.mx);
  if (o.mean > 0 && m.mean > 0)
    std::printf("# mean speedup ml_rec vs origami: %.2fx\n", o.mean / m.mean);
  if (o.med > 0 && m.med > 0)
    std::printf("# median speedup ml_rec vs origami: %.2fx\n", o.med / m.med);
  if (origami_skipped || ml_skipped)
    std::printf("# skipped (analytical rejected the problem): origami=%zu  ml_rec=%zu\n",
                origami_skipped,
                ml_skipped);
  return 0;
}
