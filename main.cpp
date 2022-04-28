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
struct FragC;
struct FragA;
struct FragB;

// catch-all definition
template<
typename MemberType, // the team policy's member_type
typename T,
unsigned M,
unsigned N,
unsigned K,
typename Kind
> struct Frag {

    // load from a view
    template<typename ViewType>
    KOKKOS_INLINE_FUNCTION Frag &operator=(const ViewType &v) {
        return *this;
    }

};

// CUDA specializations of the fragment type
#if defined(KOKKOS_ENABLE_CUDA)


template<typename MemberType> 
struct Frag<MemberType, Kokkos::Experimental::half_t, 16, 16, 16, FragC> {
    typedef FragC frag_kind;
    static constexpr unsigned M = 16;
    static constexpr unsigned N = 16;
    static constexpr unsigned K = 16;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> native_type;
    native_type frag;

    template<typename ViewType>
    KOKKOS_INLINE_FUNCTION Frag &operator=(const ViewType &v) {
        return *this;
    }

    /*! \brief
    perform C += A * B
    */
    template<typename AFragType, typename BFragType>
    KOKKOS_INLINE_FUNCTION Frag &fmma(const AFragType &a, const BFragType &b) {
        return *this;
    }

};

#endif // KOKKOS_ENABLE_CUDA



// fmma operation
// operates on an A, B, and C fragment
template<
typename MemberType,
typename AType,
typename BType,
typename CType,
std::enable_if_t<
   std::is_same<typename AType::frag_kind, FragA>::value
&& std::is_same<typename BType::frag_kind, FragB>::value
&& std::is_same<typename CType::frag_kind, FragC>::value
&& AType::M == BType::M
&& AType::N == BType::N
&& AType::K == BType::K
&& AType::M == CType::M
&& AType::N == CType::N
&& AType::K == CType::K
, bool> = true
>
KOKKOS_INLINE_FUNCTION void team_fmma(const MemberType &team_member, CType &c, AType &a, BType &b) {
    // ...
}



#if defined(KOKKOS_ENABLE_CUDA)

// fmma operation
// operates on an A, B, and C fragment
template<
typename MemberType,
typename AType,
typename BType,
typename CType
>
KOKKOS_INLINE_FUNCTION void 
team_fmma(const MemberType &team_member, CType &c, AType &a, BType &b) {
    nvcuda::wmma::mma_sync(c.frag, a.frag, b.frag, c.frag);
}


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

        printf("%d, %d\n", team_member.league_rank(), team_member.team_rank());

        Frag<MemberType, AScalar, 16, 16, 16, FragC> frag_c;

        for (size_t kb = 0; kb < A_.extent(1); kb += 16) {
            Frag<MemberType, AScalar, 16, 16, 16, FragA> frag_a;
            Frag<MemberType, AScalar, 16, 16, 16, FragB> frag_b;

            auto A_sub = Kokkos::subview(A_, Kokkos::make_pair(lim, lim+16), Kokkos::make_pair(kb, kb+16));
            auto B_sub = Kokkos::subview(B_, Kokkos::make_pair(kb, kb+16), Kokkos::make_pair(lin, lin+16));

            frag_a = A_sub;
            frag_b = B_sub;
            frag_c.fmma(frag_a, frag_b);
        }

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