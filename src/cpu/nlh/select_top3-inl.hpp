// SPDX-License-Identifier: GPL-2.0-only
// Intentionally re-included inside each Highway target namespace.
template <class D>
bool SelectTopThree64(D d, const float* distances, int self, int* indices, float* selected) {
    const hn::Rebind<std::uint32_t, D> du;
    const int lanes = int(hn::Lanes(d));
    const auto inf = hn::Set(d, 1.0e30f);
    const auto absent = hn::Set(du, 64u);
    auto a = inf, b = inf, c = inf;
    auto ia = absent, ib = absent, ic = absent;
    auto bounded = hn::FirstN(du, hn::Lanes(du));
    for (int offset = 0; offset < 64; offset += lanes) {
        auto value = hn::LoadU(d, distances + offset);
        auto index = hn::Iota(du, std::uint32_t(offset));
        // Bit comparison survives production fast-math. Negative, sentinel,
        // overflowing and NaN distances defer to the unchanged selector.
        bounded = hn::And(bounded, hn::Lt(hn::BitCast(du, value),
                            hn::Set(du, std::bit_cast<std::uint32_t>(1.0e30f))));
        value = hn::IfThenElse(hn::RebindMask(d, hn::Eq(index, hn::Set(du, std::uint32_t(self)))), inf, value);
        // Ascending offsets preserve lowest-index ties within each lane.
        const auto first = hn::Lt(value, a);
        const auto second = hn::Lt(value, b);
        const auto third = hn::Lt(value, c);
        // Select from the original ordered triple, rather than reinserting
        // a displaced item (whose equal-distance index may precede slot b).
        c = hn::IfThenElse(second, b, hn::IfThenElse(third, value, c));
        ic = hn::IfThenElse(hn::RebindMask(du, second), ib,
                           hn::IfThenElse(hn::RebindMask(du, third), index, ic));
        b = hn::IfThenElse(first, a, hn::IfThenElse(second, value, b));
        ib = hn::IfThenElse(hn::RebindMask(du, first), ia,
                           hn::IfThenElse(hn::RebindMask(du, second), index, ib));
        a = hn::IfThenElse(first, value, a);
        ia = hn::IfThenElse(hn::RebindMask(du, first), index, ia);
    }
    if (!hn::AllTrue(du, bounded)) return false;
    for (int slot = 0; slot < 3; ++slot) {
        const float best = hn::ReduceMin(d, a);
        const auto tied = hn::RebindMask(du, hn::Eq(a, hn::Set(d, best)));
        const auto best_index = hn::ReduceMin(du, hn::IfThenElse(tied, ia, absent));
        indices[slot] = int(best_index);
        selected[slot] = best;
        const auto chosen = hn::Eq(ia, hn::Set(du, best_index));
        const auto chosen_f = hn::RebindMask(d, chosen);
        a = hn::IfThenElse(chosen_f, b, a);
        ia = hn::IfThenElse(chosen, ib, ia);
        b = hn::IfThenElse(chosen_f, c, b);
        ib = hn::IfThenElse(chosen, ic, ib);
        c = hn::IfThenElse(chosen_f, inf, c);
        ic = hn::IfThenElse(chosen, absent, ic);
    }
    return true;
}
