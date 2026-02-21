#!/usr/bin/env python3
"""Generate hard/extreme synthetic AIMO-style local benchmark sets.

The output CSV contains columns:
- id
- problem
- answer
- category
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Callable


MOD_CHOICES = (99991, 99989, 65521, 60013, 50021, 100000)
LUCAS_PRIMES = (1009, 2003, 5003, 10007)

TemplateFn = Callable[[int, random.Random], tuple[str, int, str]]

_FACT_CACHE: dict[int, tuple[list[int], list[int]]] = {}


def _normalize(ans: int, modulus: int) -> int:
    if modulus <= 0:
        return ans % 100_000
    return ans % modulus


def _mat_mul(a: list[list[int]], b: list[list[int]], mod: int) -> list[list[int]]:
    n = len(a)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for k in range(n):
            aik = ai[k]
            if aik == 0:
                continue
            bk = b[k]
            for j in range(n):
                oi[j] = (oi[j] + aik * bk[j]) % mod
    return out


def _mat_pow(base: list[list[int]], exp: int, mod: int) -> list[list[int]]:
    n = len(base)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        out[i][i] = 1
    cur = [row[:] for row in base]
    e = exp
    while e > 0:
        if e & 1:
            out = _mat_mul(out, cur, mod)
        cur = _mat_mul(cur, cur, mod)
        e >>= 1
    return out


def _fib_pair(n: int, mod: int) -> tuple[int, int]:
    """Returns (F_n, F_{n+1}) under modulus."""
    if n == 0:
        return 0, 1
    a, b = _fib_pair(n >> 1, mod)
    c = (a * ((2 * b - a) % mod)) % mod
    d = (a * a + b * b) % mod
    if n & 1:
        return d, (c + d) % mod
    return c, d


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _floor_sum(n: int, m: int, a: int, b: int) -> int:
    """Compute sum_{i=0}^{n-1} floor((a*i+b)/m) in O(log m)."""
    ans = 0
    while True:
        if a >= m:
            ans += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            ans += n * (b // m)
            b %= m
        y_max = a * n + b
        if y_max < m:
            return ans
        n = y_max // m
        b = y_max % m
        a, m = m, a


def _factorials_mod_prime(p: int) -> tuple[list[int], list[int]]:
    cached = _FACT_CACHE.get(p)
    if cached is not None:
        return cached
    fact = [1] * p
    for i in range(1, p):
        fact[i] = (fact[i - 1] * i) % p
    inv_fact = [1] * p
    inv_fact[-1] = pow(fact[-1], p - 2, p)
    for i in range(p - 2, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % p
    _FACT_CACHE[p] = (fact, inv_fact)
    return fact, inv_fact


def _nck_mod_prime_small(n: int, k: int, p: int) -> int:
    if k < 0 or k > n:
        return 0
    fact, inv_fact = _factorials_mod_prime(p)
    return (((fact[n] * inv_fact[k]) % p) * inv_fact[n - k]) % p


def _lucas_binom_mod_prime(n: int, k: int, p: int) -> int:
    if k < 0 or k > n:
        return 0
    ans = 1
    nn = n
    kk = k
    while nn > 0 or kk > 0:
        ni = nn % p
        ki = kk % p
        if ki > ni:
            return 0
        ans = (ans * _nck_mod_prime_small(ni, ki, p)) % p
        nn //= p
        kk //= p
    return ans


def _template_power_mix(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    a = rng.randrange(7, 200)
    b = rng.randrange(40_000, 400_000)
    c = rng.randrange(5, 160)
    d = rng.randrange(35_000, 250_000)
    e = rng.randrange(3, 120)
    f = rng.randrange(20_000, 160_000)
    g = rng.randrange(2, 95)
    h = rng.randrange(10_000, 90_000)
    ans = (pow(a, b, mod) + pow(c, d, mod) - pow(e, f, mod) + pow(g, h, mod)) % mod
    problem = (
        f"Let S = {a}^{b} + {c}^{d} - {e}^{f} + {g}^{h}. "
        f"Compute S modulo {mod}. Return the least nonnegative residue."
    )
    return problem, _normalize(ans, mod), "number_theory_power_mix"


def _template_finite_diff_binom(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    n = rng.randrange(36, 90)
    # Keep degree above n to avoid trivial zero via finite-difference annihilation.
    t = n + rng.randrange(2, 10)
    r = rng.randrange(20, 300)
    ans = 0
    for k in range(n + 1):
        term = _comb(n, k) * pow(r + k, t, mod)
        if k & 1:
            ans -= term
        else:
            ans += term
    ans %= mod
    problem = (
        f"Define A = sum_{{k=0}}^{{{n}}} (-1)^k * C({n}, k) * ({r}+k)^{t}. "
        f"Find A modulo {mod}."
    )
    return problem, _normalize(ans, mod), "combinatorics_finite_difference"


def _template_linear_recurrence(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    p = rng.randrange(2, 130)
    q = rng.randrange(2, 130)
    u0 = rng.randrange(0, mod)
    u1 = rng.randrange(0, mod)
    n = rng.randrange(150_000, 2_000_000)

    mat = [[p % mod, q % mod], [1, 0]]
    if n == 0:
        ans = u0
    elif n == 1:
        ans = u1
    else:
        mp = _mat_pow(mat, n - 1, mod)
        ans = (mp[0][0] * u1 + mp[0][1] * u0) % mod
    problem = (
        f"Sequence (u_n) satisfies u_(n+2) = {p}*u_(n+1) + {q}*u_n with "
        f"u_0={u0}, u_1={u1}. Compute u_{n} modulo {mod}."
    )
    return problem, _normalize(ans, mod), "recurrence_linear"


def _template_fib_combo(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    n = rng.randrange(10**11, 10**12)
    k = rng.randrange(100_000, 2_000_000)
    fn, fn1 = _fib_pair(n, mod)
    fnk, _ = _fib_pair(n + k, mod)
    ans = (fn * fn1 + fnk) % mod
    problem = (
        "Let F_0=0, F_1=1, and F_(t+2)=F_(t+1)+F_t. "
        f"Compute B = F_{n}*F_{n+1} + F_{n+k} modulo {mod}."
    )
    return problem, _normalize(ans, mod), "number_theory_fibonacci"


def _template_bounded_compositions(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    k = rng.randrange(7, 13)
    bound = rng.randrange(12, 32)
    max_total = k * bound
    lower = max(k, int(0.45 * max_total))
    upper = max(lower + 1, int(0.90 * max_total))
    total = rng.randrange(lower, upper)
    ans = 0
    for j in range(k + 1):
        remaining = total - j * (bound + 1)
        if remaining < 0:
            continue
        ways = _comb(k, j) * _comb(remaining + k - 1, k - 1)
        if j & 1:
            ans -= ways
        else:
            ans += ways
    ans %= mod
    problem = (
        f"Count integer tuples (x_1,...,x_{k}) with x_1+...+x_{k}={total} and "
        f"0 <= x_i <= {bound} for all i. Give the count modulo {mod}."
    )
    return problem, _normalize(ans, mod), "combinatorics_bounded_compositions"


def _template_surjections(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    n = rng.randrange(18, 36)
    k = rng.randrange(7, min(15, n - 1))
    ans = 0
    for j in range(k + 1):
        term = _comb(k, j) * pow(k - j, n, mod)
        if j & 1:
            ans -= term
        else:
            ans += term
    ans %= mod
    problem = (
        f"How many surjective functions exist from a set of size {n} to a set of size {k}? "
        f"Return the answer modulo {mod}."
    )
    return problem, _normalize(ans, mod), "combinatorics_surjection"


def _template_poly_coeff(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    a = rng.randrange(40, 110)
    b = rng.randrange(25, 90)
    c = rng.randrange(20, 70)
    t = rng.randrange(40, 170)
    ans = 0
    # Coeff of x^t in (1+x)^a (1+x^2)^b (1+x^3)^c
    for i2 in range(a + 1):
        r1 = t - i2
        if r1 < 0:
            break
        c1 = _comb(a, i2)
        max_j = min(b, r1 // 2)
        for j in range(max_j + 1):
            r2 = r1 - 2 * j
            if r2 % 3 != 0:
                continue
            q = r2 // 3
            if q < 0 or q > c:
                continue
            ans += c1 * _comb(b, j) * _comb(c, q)
    ans %= mod
    problem = (
        f"Let P(x)=(1+x)^{a}(1+x^2)^{b}(1+x^3)^{c}. "
        f"Find the coefficient of x^{t} in P(x), modulo {mod}."
    )
    return problem, _normalize(ans, mod), "algebra_polynomial_coefficient"


def _template_matrix_trace(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    a = rng.randrange(2, 50)
    b = rng.randrange(2, 50)
    c = rng.randrange(2, 50)
    d = rng.randrange(2, 50)
    e = rng.randrange(2, 50)
    f = rng.randrange(2, 50)
    g = rng.randrange(2, 50)
    h = rng.randrange(2, 50)
    j = rng.randrange(2, 50)
    n = rng.randrange(200_000, 1_500_000)
    mat = [[a, b, c], [d, e, f], [g, h, j]]
    mp = _mat_pow([[x % mod for x in row] for row in mat], n, mod)
    ans = (mp[0][0] + mp[1][1] + mp[2][2]) % mod
    problem = (
        f"Let M=[[{a},{b},{c}],[{d},{e},{f}],[{g},{h},{j}]]. "
        f"Compute trace(M^{n}) modulo {mod}."
    )
    return problem, _normalize(ans, mod), "linear_algebra_matrix_power"


def _template_crt_composed(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    moduli = [101, 103, 107, 109]
    rng.shuffle(moduli)
    m1, m2, m3 = moduli[:3]
    r1 = rng.randrange(0, m1)
    r2 = rng.randrange(0, m2)
    r3 = rng.randrange(0, m3)

    # Solve CRT for pairwise coprime moduli.
    m = m1 * m2 * m3
    ans_crt = 0
    for r, mm in ((r1, m1), (r2, m2), (r3, m3)):
        mi = m // mm
        inv = pow(mi, -1, mm)
        ans_crt = (ans_crt + r * mi * inv) % m

    alpha = rng.randrange(2, 80)
    beta = rng.randrange(2, 200)
    ans = (ans_crt * ans_crt + alpha * ans_crt + beta) % mod
    problem = (
        f"Let x be the smallest nonnegative integer such that x ≡ {r1} (mod {m1}), "
        f"x ≡ {r2} (mod {m2}), and x ≡ {r3} (mod {m3}). "
        f"Compute x^2 + {alpha}x + {beta} modulo {mod}."
    )
    return problem, _normalize(ans, mod), "number_theory_crt"


def _template_geometry_integer(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    x1, y1 = rng.randrange(1, 80), rng.randrange(1, 80)
    x2, y2 = rng.randrange(81, 180), rng.randrange(1, 90)
    x3, y3 = rng.randrange(15, 120), rng.randrange(95, 220)

    area2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    l12 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    l23 = (x2 - x3) ** 2 + (y2 - y3) ** 2
    l31 = (x3 - x1) ** 2 + (y3 - y1) ** 2
    q = l12 + l23 + l31
    p1 = rng.randrange(3, 12)
    p2 = rng.randrange(2, 9)
    ans = (pow(area2, p1, mod) + pow(q, p2, mod) + area2 * q) % mod
    problem = (
        f"In the coordinate plane, triangle vertices are A=({x1},{y1}), B=({x2},{y2}), C=({x3},{y3}). "
        f"Let T = 2*Area(ABC) and Q = AB^2 + BC^2 + CA^2. Compute T^{p1} + Q^{p2} + TQ modulo {mod}."
    )
    return problem, _normalize(ans, mod), "geometry_coordinate"


def _template_floor_sum_affine(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    n = rng.randrange(10**9, 10**12)
    m = rng.choice((99991, 65521, 50021))
    a = rng.randrange(10**6, 10**9)
    b = rng.randrange(10**6, 10**9)
    c = rng.randrange(10**6, 10**9)
    d = rng.randrange(10**6, 10**9)
    alpha = rng.randrange(5, 400)

    s1 = _floor_sum(n, m, a, b)
    s2 = _floor_sum(n, m, c, d)
    ans = (s1 + alpha * s2 + pow(a + b + c + d, 3, mod)) % mod
    problem = (
        "For a real number t, let floor(t) denote the greatest integer <= t. "
        f"Define S = sum_(i=0)^({n}-1) floor(({a}i+{b})/{m}) + {alpha} * "
        f"sum_(i=0)^({n}-1) floor(({c}i+{d})/{m}). "
        f"Compute S + ({a}+{b}+{c}+{d})^3 modulo {mod}."
    )
    return problem, _normalize(ans, mod), "number_theory_floor_sum"


def _template_lucas_binomial_combo(i: int, rng: random.Random) -> tuple[str, int, str]:
    p = rng.choice(LUCAS_PRIMES)
    n1 = rng.randrange(10**12, 10**15)
    k1 = rng.randrange(0, n1 + 1)
    n2 = rng.randrange(10**12, 10**15)
    k2 = rng.randrange(0, n2 + 1)
    gamma = rng.randrange(2, 50)

    c1 = _lucas_binom_mod_prime(n1, k1, p)
    c2 = _lucas_binom_mod_prime(n2, k2, p)
    ans = (c1 + gamma * c2 + n1 + k2) % p
    problem = (
        f"Let p={p}. Compute T = C({n1},{k1}) + {gamma}*C({n2},{k2}) + {n1}+{k2} modulo p. "
        "Return the least nonnegative residue."
    )
    return problem, _normalize(ans, p), "number_theory_lucas_binomial"


def _template_recurrence_order4(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    coefs = [rng.randrange(1, 120) for _ in range(4)]
    init = [rng.randrange(0, mod) for _ in range(4)]
    n = rng.randrange(1_000_000, 8_000_000)

    if n < 4:
        ans = init[n]
    else:
        a0, a1, a2, a3 = [c % mod for c in coefs]
        companion = [
            [a0, a1, a2, a3],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
        mp = _mat_pow(companion, n - 3, mod)
        state = [init[3] % mod, init[2] % mod, init[1] % mod, init[0] % mod]
        ans = sum(mp[0][j] * state[j] for j in range(4)) % mod

    problem = (
        f"Define sequence (u_t) by u_(t+4)={coefs[0]}u_(t+3)+{coefs[1]}u_(t+2)+"
        f"{coefs[2]}u_(t+1)+{coefs[3]}u_t, with initial values "
        f"u_0={init[0]}, u_1={init[1]}, u_2={init[2]}, u_3={init[3]}. "
        f"Compute u_{n} modulo {mod}."
    )
    return problem, _normalize(ans, mod), "recurrence_order4"


def _template_tripartite_spanning_trees(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    n1 = rng.randrange(20, 180)
    n2 = rng.randrange(20, 180)
    n3 = rng.randrange(20, 180)
    n = n1 + n2 + n3
    # Number of spanning trees of complete tripartite graph K_{n1,n2,n3}:
    # n^(3-2) * (n-n1)^(n1-1) * (n-n2)^(n2-1) * (n-n3)^(n3-1)
    ans = n % mod
    ans = (ans * pow(n - n1, n1 - 1, mod)) % mod
    ans = (ans * pow(n - n2, n2 - 1, mod)) % mod
    ans = (ans * pow(n - n3, n3 - 1, mod)) % mod
    problem = (
        f"Let G be the complete tripartite graph K_({n1},{n2},{n3}). "
        f"How many spanning trees does G have? Return the result modulo {mod}."
    )
    return problem, _normalize(ans, mod), "combinatorics_graph_spanning_trees"


def _template_divisor_totient_weighted(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    p, q, r = 2, 3, 5
    a = rng.randrange(7, 12)
    b = rng.randrange(5, 9)
    c = rng.randrange(4, 8)
    alpha = rng.randrange(2, 9)
    beta = rng.randrange(3, 11)
    n = (p**a) * (q**b) * (r**c)

    ans = 0
    for i1 in range(a + 1):
        for i2 in range(b + 1):
            for i3 in range(c + 1):
                d = (p**i1) * (q**i2) * (r**i3)
                phi = d
                if i1 > 0:
                    phi = phi // p * (p - 1)
                if i2 > 0:
                    phi = phi // q * (q - 1)
                if i3 > 0:
                    phi = phi // r * (r - 1)
                tau = (i1 + 1) * (i2 + 1) * (i3 + 1)
                ans += phi * (tau * tau + alpha * tau + beta)
    ans %= mod

    problem = (
        f"Let N = 2^{a} * 3^{b} * 5^{c}. For each divisor d of N, define tau(d) as the number of "
        "positive divisors of d and phi(d) as Euler's totient. Compute\n"
        f"S = sum_(d|N) phi(d) * (tau(d)^2 + {alpha}*tau(d) + {beta}) modulo {mod}."
    )
    return problem, _normalize(ans, mod), "number_theory_divisor_totient_sum"


def _template_coprime_count_inclusion(i: int, rng: random.Random) -> tuple[str, int, str]:
    mod = rng.choice(MOD_CHOICES)
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    rng.shuffle(base_primes)
    chosen = base_primes[:5]
    n = rng.randrange(10**10, 10**12)
    alpha = rng.randrange(2, 80)
    beta = rng.randrange(2, 400)

    m = 1
    for p in chosen:
        m *= p

    coprime_count = 0
    for mask in range(1 << len(chosen)):
        d = 1
        bits = 0
        for j, p in enumerate(chosen):
            if (mask >> j) & 1:
                d *= p
                bits += 1
        term = n // d
        if bits & 1:
            coprime_count -= term
        else:
            coprime_count += term

    ans = (pow(coprime_count, 2, mod) + alpha * (coprime_count % mod) + beta) % mod
    problem = (
        f"Let M = {m} and N = {n}. Let C be the number of integers x with 1 <= x <= N and gcd(x,M)=1. "
        f"Compute C^2 + {alpha}C + {beta} modulo {mod}."
    )
    return problem, _normalize(ans, mod), "number_theory_coprime_count"


HARD_TEMPLATES: tuple[TemplateFn, ...] = (
    _template_power_mix,
    _template_finite_diff_binom,
    _template_linear_recurrence,
    _template_fib_combo,
    _template_bounded_compositions,
    _template_surjections,
    _template_poly_coeff,
    _template_matrix_trace,
    _template_crt_composed,
    _template_geometry_integer,
)

EXTREME_TEMPLATES: tuple[TemplateFn, ...] = HARD_TEMPLATES + (
    _template_floor_sum_affine,
    _template_lucas_binomial_combo,
    _template_recurrence_order4,
    _template_tripartite_spanning_trees,
    _template_divisor_totient_weighted,
    _template_coprime_count_inclusion,
)


def _templates_for_difficulty(difficulty: str) -> tuple[TemplateFn, ...]:
    if difficulty == "hard":
        return HARD_TEMPLATES
    if difficulty == "extreme":
        return EXTREME_TEMPLATES
    # mixed
    return EXTREME_TEMPLATES


def generate_rows(count: int, seed: int, difficulty: str = "mixed") -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    templates = _templates_for_difficulty(difficulty)
    prefix = {"hard": "hard", "extreme": "xhard", "mixed": "mix"}[difficulty]
    for i in range(1, count + 1):
        fn = templates[(i - 1) % len(templates)]
        problem, answer, category = fn(i, rng)
        rows.append(
            {
                "id": f"{prefix}_{i:03d}",
                "problem": problem,
                "answer": int(answer),
                "category": category,
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "problem", "answer", "category"])
        writer.writeheader()
        writer.writerows(rows)


def _write_unlabeled(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "problem"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "problem": row["problem"]})


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate hard/extreme synthetic local AIMO benchmark.")
    parser.add_argument("--count", type=int, default=80, help="Number of problems to generate.")
    parser.add_argument("--seed", type=int, default=20260213, help="RNG seed.")
    parser.add_argument(
        "--difficulty",
        choices=["hard", "extreme", "mixed"],
        default="mixed",
        help="Template pool to use.",
    )
    parser.add_argument(
        "--output-csv",
        default="examples/hard_synthetic_problems.csv",
        help="Output CSV path (labeled).",
    )
    parser.add_argument(
        "--output-unlabeled-csv",
        default="",
        help="Optional output CSV path with only id/problem columns.",
    )
    args = parser.parse_args()

    out_path = Path(args.output_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(
        count=max(1, args.count),
        seed=args.seed,
        difficulty=args.difficulty,
    )
    _write_rows(out_path, rows)

    unlabeled_target = args.output_unlabeled_csv.strip()
    if unlabeled_target:
        unlabeled_path = Path(unlabeled_target).expanduser()
        unlabeled_path.parent.mkdir(parents=True, exist_ok=True)
        _write_unlabeled(unlabeled_path, rows)
        print(f"Wrote unlabeled rows to {unlabeled_path}")

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Sample:")
    for row in rows[:3]:
        print(f"- {row['id']} [{row['category']}] answer={row['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
