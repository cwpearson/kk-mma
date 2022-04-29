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

  I4: To what extent do we expect a fragment user to adjust various parameters of their implementations
  for different backends? Too much and there's no point to the abstraction.

  Other Notes
  ------------

  CUDA nomenclature is m,n,k:
  matrix A is mxk, matrix B is k x n, and matrix C is mxn
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



template<typename MemberType, typename CView, typename AView, typename BView>
struct Functor {

    AView A_;
    BView B_;
    CView C_;

    typedef typename AView::non_const_value_type AScalar;
    typedef typename BView::non_const_value_type BScalar;
    typedef typename CView::non_const_value_type CScalar;

    typedef Kokkos::View<AScalar[16][16]> ASubView;
    typedef Kokkos::View<BScalar[16][16]> BSubView;
    typedef Kokkos::View<CScalar[16][16]> CSubView;


    Functor(const CView &C, const AView &A, const BView &B) :C_(C), A_(A), B_(B) {}

    KOKKOS_INLINE_FUNCTION void operator()(MemberType team_member) const {

        // which m, n fragment to operate on
        const size_t lim = team_member.league_rank() / C_.extent(1);
        const size_t lin = team_member.league_rank() % C_.extent(1);

        CSubView C_sub(C_, Kokkos::make_pair(lim, lim+16), Kokkos::make_pair(lin, lin+16));
        // auto C_sub = Kokkos::subview(C_, Kokkos::make_pair(lim, lim+16), Kokkos::make_pair(lin, lin+16));

        printf("%d, %d\n", team_member.league_rank(), team_member.team_rank());

        // this should be a CUDA frag
        Frag<MemberType, UseC, 16, 16, 16, CScalar> cuda_frag_c;
        load(team_member, cuda_frag_c, C_sub);

        for (size_t kb = 0; kb < A_.extent(1); kb += 16) {
            Frag<MemberType, UseA, 16, 16, 16, AScalar, typename AView::array_layout> frag_a;
            Frag<MemberType, UseB, 16, 16, 16, BScalar, typename BView::array_layout> frag_b;

            ASubView A_sub(A_, Kokkos::make_pair(lim, lim+16), Kokkos::make_pair(kb, kb+16));
            BSubView B_sub(B_, Kokkos::make_pair(kb, kb+16), Kokkos::make_pair(lin, lin+16));
            // auto A_sub = Kokkos::subview(A_, Kokkos::make_pair(lim, lim+16), Kokkos::make_pair(kb, kb+16));
            // auto B_sub = Kokkos::subview(B_, Kokkos::make_pair(kb, kb+16), Kokkos::make_pair(lin, lin+16));

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

    typedef Kokkos::View<Kokkos::Experimental::half_t[32][32]> AView;
    typedef Kokkos::View<Kokkos::Experimental::half_t[32][32]> BView;
    typedef Kokkos::View<float[32][32]> CView;

    AView A;
    BView B;
    CView C;


    Kokkos::View<Kokkos::Experimental::half_t[16][16]> A_sub(A, std::make_pair(0, 16), std::make_pair(0, 16));
    Kokkos::View<Kokkos::Experimental::half_t[16][16]> B_sub(B, std::make_pair(0, 16), std::make_pair(0, 16));
    Kokkos::View<float[16][16]> C_sub(C, std::make_pair(0, 160), std::make_pair(0, 16));

    typedef Kokkos::TeamPolicy<>::member_type member_type;

    const size_t mBlocks = (C.extent(0) + 16 - 1) / 16;
    const size_t nBlocks = (C.extent(1) + 16 - 1) / 16;

    const size_t leagueSize = mBlocks * nBlocks;
    const size_t teamSize = 32;

    Kokkos::TeamPolicy<> policy(leagueSize, teamSize);

    Functor<member_type, CView, AView, BView> gemm(C, A, B);

    Kokkos::parallel_for(policy, gemm);
    Kokkos::fence();


    Kokkos::finalize();

}