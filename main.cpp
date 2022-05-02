#include <Kokkos_Core.hpp>


#if defined(KOKKOS_ENABLE_CUDA)
#include <mma.h>
#elif defined(KOKKOS_ENABLE_HIP)

#endif

/*
  NVIDIA and AMD GPUs have matrix fragment types.
  KokkosKernels abstracts over those fragment types to easy development of fragment-oriented algorithms
  and automatically take advantage of underlying hardware support

  The general notion of these fragments is expected to be:
  D1) the fragment is collaboratively owned by more than one thread
  D2) the fragment has a distributed in-register representation
  D3) i and j for the i,j element(s) of the fragment each thread holds cannot be determined and is
      implementation-defined.

  S1: Fragment Types
  ------------------
  * FT cannot be a Kokkos::View<T[M][N]> because that cannot live distributed among
    the registers of many threads
  * An MxN FT should be initializable from a Kokkos::View<T[M][N]>
  * The FTs will expose team-oriented functions that will assert the team size is large enough
  * This implementation allows a single parameterized fragment-oriented implementation of various algorithms
  and different versions of that algorithm can be dispatched according to available architecture and input size

  S2: Fragment Operations
  -----------------------
  load(frag, view): load a fragment from a 2D Kokkos::view that is LayoutLeft or LayoutRight
  store(view, frag): store a fragment to ...
  fmma(c, a, b): do c += a * b

  S3: Fragment Specialization
  ---------------------------
  The abstraction allows defining fragments of any size and scalar type
  When that size and scalar type is matched, the fragment and operations will be implemented
  in terms of the hardware fragment types and intrinsics.
  Otherwise, they will be implemented as a "Fallback" type with the operations defined through standard
  Kokkos hierarchical parallelism.

  Design Issues
  --------------
  I1: A consequence of S3 and D3 is that if F1 is a fallback fragment and F2 is a CUDA fragment, it is not
  directly possible to do fmma(F1, F2, ....) because the implementation cannot know which thread holds
  which data in F2.
  A workaround is to stage F2 through scratch memory and convert it to a Fallback fragment before operating
  on it.

  I2: Fallback fmma() (and perhaps others) may require scratch memory since data from input fragments
  needs to move between threads to produce output fragments.
  How to query how much scratch is required so it can be provided?

  I3: How to handle the fact that large teams could easily work on more fragments at once?
  Right now, for CUDA fragments, the first 32 threads in a team participate and the rest
  are shut off.

  I4: related to I3, could we automatically handle larger fragments as multiple smaller fragments?

  I5: what to do if operations start to diverge, e.g. Nvidia implements fragment transpose and AMD does not.
  Do we support that operation on all backends and just be slow if it's not AMD, or do we support it on none of them?

  Other Notes
  ------------

  CUDA nomenclature is m,n,k:
  matrix A is mxk, matrix B is k x n, and matrix C is mxn

  Brainstorm
  ----------

  The problem this aims to solve is that we want a single fragment-oriented implementation that works across
  architectures.
  Problem is that Nvidia distributes fragment across 32 threads, AMD across 64 threads, and our fallback whatever we want.
  That defines the minimum team size required.
  Implementors may want to use a larger team to operate on multiple native fragments at once.
  May be best expressed as larger fragment types distributed across the team
  How to decide whether the larger fragment should be stacked up on one group of threads, or distributed across multiple groups?

  * Larger team containing more independent fragments (to increase occupancy)
    * "Vector Fragment"
        * CUDA / AMD: one fragment per hardware group
        * Fallback: arbitrarily size "hardware groups" and do the same
  * Larger team with a larger fragment (shared memory, etc)
    one fragment per hardware group
    different type because operations are implemented differently

  Perhaps the # of independent fragments depends on the team size, and using a larger fragment requires a larger team size

  Perhaps the user specifies a fragment size, and then asks how many threads per team are needed
  Perhaps the abstraction is wrong

  Want to be able to use a team to operate on multiple fragments at once
  Certain operations could be optimized by having a larger team on a larger fragment
  The way to express this is to merge the fragments into a larger one and issue a supported operation
  The larger team distributes the fragments throughout the team (one/more fragment per 32 threads)

  Vertically stack P fragments
  v_stack(f1<UseA, M1, N, K>, f2<UseA, M2, N, K>, ...) -> Frag<UseA, M1+M2+..., N, K>
  v_stack(f1<UseB, M, N, K1>, f2<UseB, M, N, K2>, ...) -> Frag<UseB, M, N, K1+K2+...>
  v_stack(f1<UseC, M1, N, K>, f2<UseC, M2, N, K>, ...) -> Frag<UseC, M1+M2+..., N, K>

  Horizontally stack fragments
  h_stack(f1<UseA, M, N, K1>, f2<UseA, M, N, K2>) -> Frag<UseA, 2, N, K1+K2+...>
  h_stack(f1<UseB, M, N1, K>, f2<UseB, M, N2, K>) -> Frag<UseB, M, N1+N2+..., K>
  h_stack(f1<UseC, M, N1, K>, f2<UseC, M, N2, K>) -> Frag<UseC, M, N1+N2+..., K>

  Vertically split fragments
  v_split<M1, M2, ...>(f<UseA, M1+M2+..., N, K>) -> f1<UseA, M1, N, K>, f2<UseA, M2, N, K>, f3<UseA, ..., N, K>

  Horizontally split fragments



  To-Do
  -----
  [ ] loading from a smaller subview zeros the rest of the fragment
  [ ] correct API for shared memory requirements of fragment operations
    [ ] for now, a partner function that takes the same arguments as the operation but returns shared memory requirements
*/


// Where the fragment comes from; matrix A, B, or C in GEMM
struct UseC;
struct UseA;
struct UseB;

// Every FallbackFrag shall inherit from this so it's easy to use
// std::is_base_of to determine what type is a FallbackFrag
struct FallbackBase {};

/*! \brief The default fragment type when no special hardware support is available

    Impedence mismatch due to dynamic team sizes, but we want to embed
    static amount of data in the fragment struct for each thread.
    To prevent the size of the fragment from blowing up too much,
    we assume a team size of at least 32.

    Nvidia and AMD gpus do not have this problem, since their architectures have fixed
    wavefront sizes (e.g. 32,64) so they know how much data each fragment needs to hold
*/
template<
typename Use,
unsigned m,
unsigned n,
unsigned k,
typename T
>
struct FallbackFrag : public FallbackBase {
    typedef Use use_type;
    static constexpr unsigned TEAM_SZ_AT_LEAST = 32;
    static constexpr unsigned data_per_thread = (m * n * k + TEAM_SZ_AT_LEAST - 1) / TEAM_SZ_AT_LEAST;
    T data[data_per_thread];
};

/// \brief Frags that are FallbackBase are multiplied with some Kokkos-style implementation
template<
typename MemberType,
typename CType, typename AType, typename BType,
std::enable_if_t<
   std::is_base_of<FallbackBase, CType>::value
&& std::is_base_of<FallbackBase, AType>::value
&& std::is_base_of<FallbackBase, BType>::value
,bool > = true
>
KOKKOS_INLINE_FUNCTION void fmma(const MemberType &team_member, CType &c, const AType &a, const BType &b) {
    // some magic to shuffle data around between threads and produce the correct result
    // ...
}



/*! by default, a Frag is a FallbackFrag (no special device support)

    Use = UseA or UseB should specify a layout equivalent to the Kokkos::view they
    will be loaded from and stored to
*/
template<
typename MemberType,
typename Use,
unsigned m,
unsigned n,
unsigned k,
typename T,
typename Layout = void
> struct Frag : public FallbackFrag<Use, m, n, k, T> {};


// load() for FallbackFrags
template<
typename MemberType,
typename FType,
typename ViewType,
std::enable_if_t<
   std::is_base_of<FallbackBase, FType>::value
,bool > = true
>
KOKKOS_INLINE_FUNCTION void load(const MemberType &team_member, FType &f, const ViewType &view) {
    // each thread loads the correct data elements
    // ...
}

// If CUDA is available, we will have CUDA specializations of certain fragment types
#if defined(KOKKOS_ENABLE_CUDA)

// convert Kokkos layouts to corresponding wmma layouts
template<typename Layout>
struct to_wmma_layout;
template<>
struct to_wmma_layout<Kokkos::LayoutLeft> {
    typedef nvcuda::wmma::row_major type;
};
template<>
struct to_wmma_layout<Kokkos::LayoutRight> {
    typedef nvcuda::wmma::col_major type;
};
template<>
struct to_wmma_layout<void> {
    typedef void type;
};

// Tag to discriminate all classes generated from the CudaFrag class template.
// They should all inherit from this, so it's easy to check if they have hardware support
struct CudaBase {};

template<
typename Use,
unsigned m,
unsigned n,
unsigned k,
typename T,
typename Layout = void
>
// struct CudaFrag : public CudaBase {};
struct CudaFrag;

/*!
CudaFrag is a wrapper that translates Kokkos types to the CUDA native types
We also specialize Frag to use the supported CudaFrag when available

CudaFrag is only defined for supported CUDA types (otherwise it is just declared).
This varies according to architecture.

CudaFrag shall all inherit from CudaBase, which provides no members but allows us to use
std::is_base_of to easily check if a type is a CudaFrag

*/

// wrap the native CUDA type (allows Nvidia & AMD to have the same interface for Kokkos Kernels)
template<>
struct CudaFrag<UseC, 16,16,16, float, void> : public CudaBase {
    typedef UseC use_type;
    typedef nvcuda::wmma::fragment< nvcuda::wmma::accumulator, 16, 16, 16, float> native_type;
    native_type native_frag;
};

// specialize this particular Frag type to use the corresponding wrapped CUDA type
template<typename MemberType>
struct Frag<MemberType, UseC, 16, 16, 16, float, void> : public CudaFrag<UseC, 16,16,16, float, void> {};

template<typename Layout>
struct CudaFrag<UseA, 16,16,16, Kokkos::Experimental::half_t, Layout> : public CudaBase {
    typedef UseA use_type;
    typedef nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, 16, 16, 16, half, typename to_wmma_layout<Layout>::type
    > native_type;
    native_type native_frag;
};

template<typename MemberType, typename Layout>
struct Frag<MemberType, UseA, 16, 16, 16, Kokkos::Experimental::half_t, Layout>
 : public CudaFrag<UseA, 16,16,16, Kokkos::Experimental::half_t, Layout> {};

template<typename Layout>
struct CudaFrag<UseB, 16,16,16, Kokkos::Experimental::half_t, Layout>  : public CudaBase {
    typedef UseB use_type;
    typedef nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, 16, 16, 16, half, typename to_wmma_layout<Layout>::type
    > native_type;
    native_type native_frag;
};

template<typename MemberType, typename Layout>
struct Frag<MemberType, UseB, 16, 16, 16, Kokkos::Experimental::half_t, Layout>
 : public CudaFrag<UseB, 16,16,16, Kokkos::Experimental::half_t, Layout> {};

/*! specialize the Frag interface for the CudaFrag specializations.
    These should hopefully be thin wrappers around nvuda::wmma functions
*/
template<
typename MemberType,
typename CType, typename AType, typename BType,
std::enable_if_t<
   std::is_base_of<CudaBase, CType>::value
&& std::is_base_of<CudaBase, AType>::value
&& std::is_base_of<CudaBase, BType>::value
,bool > = true
>
KOKKOS_INLINE_FUNCTION void fmma(const MemberType &team_member, CType &c, const AType &a, const BType &b) {
    if (team_member.team_rank() < 32) {
        nvcuda::wmma::mma_sync(c.native_frag, a.native_frag, b.native_frag, c.native_frag);
    }
}

// specialize load for non-accumulator fragment types
template<
typename MemberType,
typename FType,
typename ViewType,
std::enable_if_t<
   std::is_base_of<CudaBase, FType>::value
&& !std::is_same<typename FType::use_type, UseC>::value
,bool > = true
>
KOKKOS_INLINE_FUNCTION void load(const MemberType &team_member, FType &f, const ViewType &view) {
    if (team_member.team_rank() < 32) {
        nvcuda::wmma::load_matrix_sync(f.native_frag, nullptr, 0);
    }
}

// specialize load for accumulator fragment type
template<
typename MemberType,
typename FType,
typename ViewType,
std::enable_if_t<
   std::is_base_of<CudaBase, FType>::value
&& std::is_same<typename FType::use_type, UseC>::value
,bool > = true
>
KOKKOS_INLINE_FUNCTION void load(const MemberType &team_member, FType &f, const ViewType &view) {
    if (team_member.team_rank() < 32) {
        // TODO, this should be a static specialization?
        if (std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value) {
           nvcuda::wmma::load_matrix_sync(f.native_frag, nullptr, 0, nvcuda::wmma::layout_t::mem_row_major);
        } else if (std::is_same<typename ViewType::array_layout, Kokkos::LayoutLeft>::value) {
            nvcuda::wmma::load_matrix_sync(f.native_frag, nullptr, 0, nvcuda::wmma::layout_t::mem_col_major);
        }
    }
}

// specialize store for accumulator fragment type
template<
typename MemberType,
typename FType,
typename ViewType,
std::enable_if_t<
   std::is_base_of<CudaBase, FType>::value
&& std::is_same<typename FType::use_type, UseC>::value
,bool > = true
>
KOKKOS_INLINE_FUNCTION void store(const MemberType &team_member, ViewType &view, const FType &f) {
    if (team_member.team_rank() < 32) {
        // TODO, this should be a static specialization?
        if (std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value) {
            nvcuda::wmma::store_matrix_sync(nullptr, f.native_frag, 0, nvcuda::wmma::layout_t::mem_row_major);
        } else if (std::is_same<typename ViewType::array_layout, Kokkos::LayoutLeft>::value) {
            nvcuda::wmma::store_matrix_sync(nullptr, f.native_frag, 0, nvcuda::wmma::layout_t::mem_col_major);
        }
    }
}

#elif defined(KOKKOS_ENABLE_HIP)

// do some similar wrapping of the HIP fragment types and functions
// ...

#endif // KOKKOS_ENABLE_CUDA, KOKKOS_ENABLE_HIP



template<typename MemberType, typename CView, typename AView, typename BView,
unsigned M, unsigned N, unsigned K>
struct Functor {

    AView A_;
    BView B_;
    CView C_;

    typedef typename AView::non_const_value_type AScalar;
    typedef typename BView::non_const_value_type BScalar;
    typedef typename CView::non_const_value_type CScalar;

    typedef Kokkos::View<AScalar[M][K]> ASubView;
    typedef Kokkos::View<BScalar[K][N]> BSubView;
    typedef Kokkos::View<CScalar[M][N]> CSubView;


    Functor(const CView &C, const AView &A, const BView &B) :C_(C), A_(A), B_(B) {}

    KOKKOS_INLINE_FUNCTION void operator()(MemberType team_member) const {

        // which m, n fragment to operate on
        const size_t lim = team_member.league_rank() / C_.extent(1);
        const size_t lin = team_member.league_rank() % C_.extent(1);

        CSubView C_sub(C_, Kokkos::make_pair(lim, lim+M), Kokkos::make_pair(lin, lin+N));
        // auto C_sub = Kokkos::subview(C_, Kokkos::make_pair(lim, lim+M), Kokkos::make_pair(lin, lin+N));

        printf("%d, %d\n", team_member.league_rank(), team_member.team_rank());

        // this should be a CUDA frag
        Frag<MemberType, UseC, M, N, K, CScalar> cuda_frag_c;
        load(team_member, cuda_frag_c, C_sub);

        for (size_t kb = 0; kb < A_.extent(1); kb += K) {
            Frag<MemberType, UseA, M, N, K, AScalar, typename AView::array_layout> frag_a;
            Frag<MemberType, UseB, M, N, K, BScalar, typename BView::array_layout> frag_b;

            ASubView A_sub(A_, Kokkos::make_pair(lim, lim+M), Kokkos::make_pair(kb, kb+K));
            BSubView B_sub(B_, Kokkos::make_pair(kb, kb+K), Kokkos::make_pair(lin, lin+N));
            // auto A_sub = Kokkos::subview(A_, Kokkos::make_pair(lim, lim+M), Kokkos::make_pair(kb, kb+K));
            // auto B_sub = Kokkos::subview(B_, Kokkos::make_pair(kb, kb+K), Kokkos::make_pair(lin, lin+N));

            load(team_member, frag_a, A_sub);
            load(team_member, frag_b, B_sub);
            fmma(team_member, cuda_frag_c, frag_a, frag_b);
        }

        store(team_member, C_sub, cuda_frag_c);

#if 0
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 25), 
            [&] (int &i) {
                printf("%d, %d, %d\n", team_member.league_rank(), team_member.team_rank(), i);
            }
        );
#endif
    }
};

int main(int argc, char **argv) {

    Kokkos::initialize(argc, argv);

    // input matrix
    typedef Kokkos::View<Kokkos::Experimental::half_t[32][32]> AView;
    typedef Kokkos::View<Kokkos::Experimental::half_t[32][32]> BView;
    typedef Kokkos::View<float[32][32]> CView;
    AView A;
    BView B;
    CView C;

    // policy
    typedef Kokkos::TeamPolicy<>::member_type member_type;

    // define M,N,K appropriate for your architecture
    constexpr unsigned M = 16;
    constexpr unsigned N = 16;
    constexpr unsigned K = 16;

    // policy configuration based on matrix size and architecture
    const size_t mBlocks = (C.extent(0) + M - 1) / M;
    const size_t nBlocks = (C.extent(1) + N - 1) / N;
    const size_t leagueSize = mBlocks * nBlocks;
    const size_t teamSize = 32;
    Kokkos::TeamPolicy<> policy(leagueSize, teamSize);

    // run the multiplication
    Functor<member_type, CView, AView, BView, M, N, K> gemm(C, A, B);
    Kokkos::parallel_for(policy, gemm);
    Kokkos::fence();


    Kokkos::finalize();
}