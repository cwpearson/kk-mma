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
  1) the fragment is collaboratively owned by more than one thread
  2) the fragment has an in-register representation
  3) no thread may query which element(s) of the fragment it holds

  Fragment Types
  --------------
  * FT cannot be a Kokkos::View<T[M][N]> because that cannot live distributed among
    the registers of many threads
  * An MxN FT should be initializable from a Kokkos::View<T[M][N]>
  * The FTs will expose team-oriented functions that will assert the team size is large enough

  a matrix-multiply accumulate is basically a gemm with alpha and beta 1

  C = 1*A*B + 1*C

  it's easy to support alpha > 1 by just using the cooperative threads to scale either A or B
  before the operation
  Therefore, for various fixed-size views and types, we can provide hardware specializations

  CUDA nomenclature is m,n,k:
  matrix A is mxk, matrix B is k x n, and matrix C is mxn
*/

// matrix A, B, or C in gemm
enum class FragKind {A, B, C};

// tags
struct UseC;
struct UseA;
struct UseB;


// for std::is_base_of 
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

/// \brief Frags that are FallbackBase can be multiplied
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



// by default, Frag is a fallback frag (no special device support)
template<
typename MemberType,
typename Use,
unsigned m,
unsigned n,
unsigned k,
typename T
> struct Frag : public FallbackFrag<Use, m, n, k, T> {};


// how to load fallback fragments
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

// CUDA specializations of the fragment type
#if defined(KOKKOS_ENABLE_CUDA)

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
struct CudaFrag<UseC, 16,16,16, float> : public CudaBase {
    typedef UseC use_type;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> native_type;
    native_type native_frag;
};

// specialize this particular Fragment type to use the wrapped CUDA type
template<typename MemberType>
struct Frag<MemberType, UseC, 16, 16, 16, float> : public CudaFrag<UseC, 16,16,16, float> {};

template<>
struct CudaFrag<UseA, 16,16,16, Kokkos::Experimental::half_t> : public CudaBase {
    typedef UseA use_type;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> native_type;
    native_type native_frag;
};

template<typename MemberType>
struct Frag<MemberType, UseA, 16, 16, 16, Kokkos::Experimental::half_t> : public CudaFrag<UseA, 16,16,16, Kokkos::Experimental::half_t> {};

template<>
struct CudaFrag<UseB, 16,16,16, Kokkos::Experimental::half_t>  : public CudaBase {
    typedef UseB use_type;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> native_type;
    native_type native_frag;
};

template<typename MemberType>
struct Frag<MemberType, UseB, 16, 16, 16, Kokkos::Experimental::half_t> : public CudaFrag<UseB, 16,16,16, Kokkos::Experimental::half_t> {};


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

#endif // KOKKOS_ENABLE_CUDA



#if defined(KOKKOS_ENABLE_CUDA)

#elif defined(KOKKOS_ENABLE_HIP)

#endif 


template<typename MemberType, typename CView, typename AView, typename BView>
struct Functor {

    AView A_;
    BView B_;
    CView C_;

    typedef typename AView::non_const_value_type AScalar;
    typedef typename BView::non_const_value_type BScalar;
    typedef typename CView::non_const_value_type CScalar;

    Functor(const CView &C, const AView &A, const BView &B) :C_(C), A_(A), B_(B) {}

    KOKKOS_INLINE_FUNCTION void operator()(MemberType team_member) const {

        // which m, n fragment to operate on
        const size_t lim = team_member.league_rank() / C_.extent(1);
        const size_t lin = team_member.league_rank() % C_.extent(1);

        auto C_sub = Kokkos::subview(C_, Kokkos::make_pair(lim, lim+16), Kokkos::make_pair(lin, lin+16));

        printf("%d, %d\n", team_member.league_rank(), team_member.team_rank());

        // this should be a CUDA frag
        Frag<MemberType, UseC, 16, 16, 16, CScalar> cuda_frag_c;
        load(team_member, cuda_frag_c, C_sub);

        // this should be a Fallback frag (double not supported in CUDA)
        Frag<MemberType, UseC, 16,16,16, double> fall_frag_c;
        load(team_member, fall_frag_c, C_sub);

        for (size_t kb = 0; kb < A_.extent(1); kb += 16) {
            Frag<MemberType, UseA, 16, 16, 16, AScalar> frag_a;
            Frag<MemberType, UseB, 16, 16, 16, BScalar> frag_b;

            auto A_sub = Kokkos::subview(A_, Kokkos::make_pair(lim, lim+16), Kokkos::make_pair(kb, kb+16));
            auto B_sub = Kokkos::subview(B_, Kokkos::make_pair(kb, kb+16), Kokkos::make_pair(lin, lin+16));

            load(team_member, frag_a, A_sub);
            load(team_member, frag_b, B_sub);
            fmma(team_member, cuda_frag_c, frag_a, frag_b);
        }

        store(C_sub, cuda_frag_c);

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