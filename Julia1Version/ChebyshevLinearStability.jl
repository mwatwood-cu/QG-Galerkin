using LinearAlgebra
#------------------------------------------------------------------------------
#
# The following functions create a Chebyshev differentiation matrix and grid
#
#------------------------------------------------------------------------------

function chebyshevZ(sizeN)
    N = sizeN - 1
    elements = zeros(sizeN)
    for j = 0:N
        elements[N+1-j] = (cos(pi*j/N) + 1.0)/2.0;
    end
    return elements
end

function myC(k,N)
    if k==0
        c=2
    elseif k==N
        c=2
    else
        c=1
    end
    return c
end

function chebyshevDM(sizeN)
    N = sizeN
    D = zeros(N,N)
    N = N - 1 # convention
    x = cos.(pi*(0:N)/N)
    for ll=0:N
        for jj=1:N-1
            if jj==ll 
                D[jj+1,ll+1] = -x[jj+1]/(2*sin(jj*pi/N)^2)
            else
                D[jj+1,ll+1] = -.5*myC(jj,N)*(-1)^(jj+ll)/
                 (myC(ll,N)*sin((jj+ll)*pi/(2*N))*sin((jj-ll)*pi/(2*N)))
            end
        end
    end

    jj=0
    for ll=1:N-1
        D[jj+1,ll+1] = -(-1)^(jj+ll)/
         (sin((jj+ll)*pi/(2*N))*sin((jj-ll)*pi/(2*N)))
    end
    jj=N
    for ll=1:N-1
        D[jj+1,ll+1] = -(-1)^(jj+ll)/
         (sin((jj+ll)*pi/(2*N))*sin((jj-ll)*pi/(2*N)))
    end
    
    D[1,1] = (2*N^2+1)/6
    D[1,end] = .5*(-1)^N
    D[end,1] = -D[1,end]
    D[end,end] = -D[1,1]
    
    D = 2*rot180(D) # Our grid is 0 -> 1, not 1 -> -1.
    return D
end


#------------------------------------------------------------------------------
#
# The following functions generate the L, U, and Q matrices for the Chebyshev-
# collocation linear stability problem
#
#------------------------------------------------------------------------------

function L_Cheb(sizeN::Int64,S::Function)
    z_k = chebyshevZ(sizeN)
    D = chebyshevDM(sizeN)
    L = zeros(sizeN,sizeN)
    for kk=1:sizeN
        L[kk,:] = S(z_k[kk])*D[kk,:]
    end
    L =-D*L
    return L
end

function U_Cheb(sizeN::Int64, dq::Function, dvt_plus, dvt_minus, u_zero, S::Function)
# This function computes the Chebyshev-collocation approximation to 
# u_bar using PV and surface buoyancy gradients, then returns the U matrix
    z_k = chebyshevZ(sizeN)
    D = chebyshevDM(sizeN)
    L = L_Cheb(sizeN,S)
    dq_vec = zeros(sizeN)
    for kk=1:sizeN
        dq_vec[kk] = dq(z_k[kk])
    end

    # Solving for u_bar is underdetermined
    # We simply append a condition that u_bar(0) = u_zero
    A = zeros(sizeN+1,sizeN)
    A[1,:] =-D[1,:]
    A[2:sizeN-1,:] = L[2:sizeN-1,:]
    A[sizeN,:] =-D[sizeN,:]
    A[sizeN+1,1] = 1.0
    b = zeros(sizeN+1)
    b[1] = dvt_minus
    b[2:sizeN-1] = dq_vec[2:sizeN-1]
    b[sizeN] = dvt_plus
    b[sizeN+1] = u_zero
    u = A\b
    fullMatrix = Diagonal(u)
    return fullMatrix
end

function Q_Cheb(sizeN, dq::Function)
    zk = chebyshevZ(sizeN)
    dq_vec = zeros(sizeN)
    for kk=1:sizeN
        dq_vec[kk] = dq(zk[kk])
    end
    fullMatrix = Diagonal(dq_vec)
    return fullMatrix
end

#------------------------------------------------------------------------------
#
# The following function solves the Chebsyhev-collocation linear stability
# problem
#
#------------------------------------------------------------------------------

function growthRateEigenValues_Cheb(sizeOfMatrix, beta_value, ky, kxValues, S::Function, dq::Function, dvt_plus, dvt_minus, u_zero)
    U = U_Cheb(sizeOfMatrix, dq, dvt_plus, dvt_minus, u_zero, S)
    L = L_Cheb(sizeOfMatrix, S)
    Q = Q_Cheb(sizeOfMatrix, dq)
    D = chebyshevDM(sizeOfMatrix)
    #println("FD Matrices Computed")
    
    steps = length(kxValues)
    
    growthRates = zeros(steps)
    waveSpeeds = zeros(steps)
    eigenvectors = []
    for i=1:steps
        #println()
        kx = kxValues[i]
        lhs = U*(Diagonal(zeros(sizeOfMatrix).+(kx^2+ky^2))+L) - (Q+Diagonal(zeros(sizeOfMatrix).+beta_value))
        rhs = Diagonal(zeros(sizeOfMatrix).+(kx^2+ky^2))+L
        lhs[1,:] = U[1,1]*D[1,:]
        lhs[1,1] = lhs[1,1] + dvt_minus
        rhs[1,:] = D[1,:]
        lhs[end,:] = U[end,end]*D[end,:]
        lhs[end,end] = lhs[end,end] + dvt_plus
        rhs[end,:] = D[end,:]
        #println(lhs)
        eigenObject = eigen(lhs, rhs)
        max_im, max_ind = findmax(imag(eigenObject.values))
        #println(eigenObject[:vectors])
        append!(eigenvectors, [eigenObject.vectors[:,max_ind]])
        waveSpeeds[i] = real(eigenObject.values[max_ind])
        growthRates[i] = max_im*kx
    end
    return growthRates, eigenvectors, waveSpeeds
end

function createFourStabilityPlots_Cheb(sizeOfMatrix, S_function, growthRates, eigenvectors, waveSpeeds, kx_values, ky_value)
    D = chebyshevDM(sizeOfMatrix)
    L = L_Cheb(sizeOfMatrix, S_function)
    
    maxIndex = argmax(growthRates)
    maxKx = kx_values[maxIndex]
    #println("Max growth rate at kx=$maxKx")

    highVector = eigenvectors[maxIndex]
    vectorSize = length(highVector)
    println(size(Diagonal(zeros(sizeOfMatrix).+(maxKx^2+ky^2))))
    psi_coeffs = (-Diagonal(zeros(sizeOfMatrix).+(maxKx^2+ky^2))+L)\highVector
    
    heights = collect(LinRange(0,1,100))
    psi_angles = zeros(length(heights))
    psi_amps = zeros(length(heights))
    for i=1:length(heights)
        psi_val = galerkin_p_psi_print(sizeOfMatrix, psi_coeffs, heights[i])
        psi_angles[i] = angle(psi_val)
        psi_amps[i] = abs(psi_val)
    end

    fig, axes = subplots(2,2)
    ax = axes[1,1]
    ax[:plot](kx_values, growthRates, label="N = $matrixSize")
    ax[:legend](loc="upper right")
    ax = axes[2,1]
    ax[:plot](kx_values, waveSpeeds, "b-")
    
    ax = axes[1,2]
    ax[:plot](psi_angles .- psi_angles[1], heights)

    ax = axes[2,2]
    ax[:plot](psi_amps/maximum(psi_amps), heights)
end
