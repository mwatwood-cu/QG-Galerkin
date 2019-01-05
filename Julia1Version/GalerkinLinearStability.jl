using FastGaussQuadrature
using Jacobi
using LinearAlgebra

#------------------------------------------------------------------------------
#
# The following functions generate the M, B, and L matrices, and evaluate a
# Galerkin approximation at a particular spatial location
#
#------------------------------------------------------------------------------

function b_n(n::Int64)
    numerator = -n*(n+1)
    denominator = (n+2)*(n+3)
    return numerator/denominator
end

function M_ij(sizeN::Int64)
    fullMatrix = zeros(sizeN,sizeN)
    for i=0:sizeN-1
        for j=0:sizeN-1
            if i==j
                fullMatrix[i+1, j+1] = 2/(2*i+1) + (b_n(i)*b_n(i))*(2/(2*i+5))
            elseif i==j+2
                fullMatrix[i+1, j+1] = b_n(j)*(2/(2*j+5))
            elseif i==j-2
                fullMatrix[i+1, j+1] = b_n(i)*(2/(2*j+1))
            end
        end
    end
    return fullMatrix/2
end

function B_ij(sizeN::Int64)
    fullMatrix = zeros(sizeN,sizeN)
    for i=0:sizeN-1
        for j=0:sizeN-1
            if i==j
                fullMatrix[i+1,j+1]=2/(2*i+1)
            elseif i==j-2
                fullMatrix[i+1,j+1]=b_n(i)*2/(2*i+5)
            end
        end
    end
    return fullMatrix/2
end

function L_ij(sizeN::Int64, S::Function)
    fullArray = zeros(sizeN,sizeN)
    values, weights = gausslegendre(nodePoints)
    for i=0:sizeN-1
        for j=i:sizeN-1
            sumTotal = 0
            for k=0:nodePoints-1
                dp_i = 2.0*(djacobi(values[k+1],i,0,0)+b_n(i)*djacobi(values[k+1],i+2,0,0))
                dp_j = 2.0*(djacobi(values[k+1],j,0,0)+b_n(j)*djacobi(values[k+1],j+2,0,0))
                newPiece = weights[k+1]*S((values[k+1]+1.0)/2.0)*dp_i*dp_j
                sumTotal = sumTotal + newPiece
            end
            fullArray[i+1,j+1] = .5*sumTotal
            fullArray[j+1,i+1] = .5*sumTotal
        end
    end
    return fullArray
end

function galerkin_p_psi_print(sizeN::Int64, coefficients, value)
    sumTotal = 0
    for i=sizeN-1:-1:0
        p = jacobi(2.0*value -1., i, 0,0)+b_n(i)*jacobi(2.0*value -1.,i+2,0,0)
        sumTotal = sumTotal + coefficients[i+1]*p
    end
    return sumTotal
end

#------------------------------------------------------------------------------
#
# The following functions generate the U and Q matrices for the Galerkin linear
# stability problem
#
#------------------------------------------------------------------------------

function U_ij(sizeN::Int64, u_galerkin_coefficients)
    fullArray = zeros(sizeN,sizeN)
    values, weights = gausslegendre(nodePoints)
    for i=0:sizeN-1
        for j=0:sizeN-1
            sumTotal = 0
            for k=0:nodePoints-1
                p_psi = jacobi(values[k+1],i,0,0)+b_n(i)*jacobi(values[k+1],i+2,0,0)
                p_q = jacobi(values[k+1],j,0,0)
                u_galerk = u_galerkin(size, u_galerkin_coefficients, (values[k+1]+1.)/2.)
                sumTotal = sumTotal + weights[k+1]*p_psi*p_q*u_galerk
            end
            fullArray[i+1, j+1] = .5*sumTotal
        end
    end
    return fullArray
end

function Q_ij(sizeN::Int64, dq_galerkin_coefficients)
    fullArray = zeros(sizeN,sizeN)
    values, weights = gausslegendre(nodePoints)
    for i=0:sizeN-1
        for j=i:sizeN-1
            sumTotal = 0
            for k=0:nodePoints-1
                dq_galerk = dq_galerkin(size, dq_galerkin_coefficients, (values[k+1]+1.)/2.)
                p_psi_i = jacobi(values[k+1],i,0,0)+b_n(i)*jacobi(values[k+1],i+2,0,0)
                p_psi_j = jacobi(values[k+1],j,0,0)+b_n(j)*jacobi(values[k+1],j+2,0,0)
                sumTotal = sumTotal + weights[k+1]*dq_galerk*p_psi_i*p_psi_j
            end
            fullArray[i+1, j+1] = .5*sumTotal
            fullArray[j+1, i+1] = .5*sumTotal
        end
    end
    return fullArray
end

#------------------------------------------------------------------------------
#
# The following functions generate various auxiliary vectors and approximations
# for the Galerkin linear stability problem
#
#------------------------------------------------------------------------------

function dq_galerkin_coefficients(sizeN::Int64, dq_bar::Function)
    fullArray = zeros(sizeN)
    values, weights = gausslegendre(nodePoints)
    for i =0:sizeN-1
        sumTotal = 0
        for k=0:nodePoints-1
            p_n = jacobi(values[k+1], i,0,0)
            sumTotal = sumTotal + weights[k+1]*p_n*dq_bar((values[k+1]+1.)/2.)
        end
        fullArray[i+1] = sumTotal*(2*i+1)/2
    end
    return fullArray
end

function dq_galerkin(sizeN::Int64, coefficients, value::Float64)
    sumTotal = 0
    for i=0:sizeN-1
        sumTotal = sumTotal + coefficients[i+1]*jacobi(2.0*value-1.,i,0,0)
    end
    return sumTotal
end

function p_psi_function(sizeN::Int64, value::Float64)
    fullArray = zeros(sizeN)
    for i=0:sizeN-1
        fullArray[i+1] = jacobi(2.0*value-1.,i,0,0)+b_n(i)*jacobi(2.0*value-1.,i+2,0,0)
    end
    return fullArray
end

function p_q_function(sizeN::Int64, value::Float64) # We shouldn't need this function
    fullArray = zeros(sizeN)
    for i=0:sizeN-1
        fullArray[i+1] = jacobi(2.0*value-1.,i,0,0)
    end
    return fullArray
end

function dy_vartheta_bar(S::Function, du_bar::Function, value::Float64)
    return -S(value)*du_bar(value)
end

function dy_psi(u_bar::Function, value::Float64)
    return -u_bar(value)
end

function dy_psi_galerkin(sizeN::Int64, u_bar_coefficients, value::Float64)
    return -u_galerkin(sizeN, u_bar_coefficients, value)
end

function u_galerkin_coefficients(sizeN::Int64, dq_coefficients, dvt_plus, dvt_minus, first_u_coefficient, du_bar::Function, S::Function, LMatrix, BMatrix)
    r_piece1 = BMatrix*dq_coefficients
    r_piece2 = dvt_plus*p_psi_function(sizeN,1.0)
    r_piece3 = dvt_minus*p_psi_function(sizeN,0.0)
    rhs = r_piece1-r_piece2+r_piece3
    lhs = LMatrix[2:sizeN,2:sizeN]
    rhs = rhs[2:sizeN]
    coefficients = lhs \ rhs
    return prepend!(coefficients,[first_u_coefficient])
end

function u_galerkin(sizeN::Int64, coefficients, value)
    sumTotal = 0
    for i=0:sizeN-1
        nextCoe = coefficients[i+1]
        sumTotal = sumTotal + coefficients[i+1]*(jacobi(2.0*value -1., i, 0,0)+b_n(i)*jacobi(2.0*value -1.,i+2,0,0))
    end
    return sumTotal
end

#------------------------------------------------------------------------------
#
# The following functions implement the Galerkin linear stability calculation
# for aribitrary background/equilibrium profiles
#
#------------------------------------------------------------------------------

function psi(sizeN::Int64, kx, ky, M_matrix, L_matrix, value)
    first_piece = (kx^2+ky^2)*M_matrix+L_matrix
    second_piece = p_psi_function(sizeN, value)
    return first_piece \ second_piece
end

function computeMatricesForStabilityAnalysis(sizeOfMatrix, S_function, dq_function, du_function,dvt_plus, dvt_minus, first_coefficient)
    B = B_ij(sizeOfMatrix)
    L = L_ij(sizeOfMatrix, S_function)
    M = M_ij(sizeOfMatrix)
    dq_coefficients = dq_galerkin_coefficients(sizeOfMatrix, dq_function)
    u_coefficients = u_galerkin_coefficients(sizeOfMatrix, dq_coefficients, dvt_plus, dvt_minus, first_coefficient, du_function, S_function, L, B)

    U = U_ij(sizeOfMatrix, u_coefficients)
    Q = Q_ij(sizeOfMatrix, dq_coefficients)
    #println("Matrices Computed")
    return (B,L,M,U,Q, u_coefficients, dq_coefficients)
end

function getSimplifiedStabilityEigenObjectsOfKx(sizeOfMatrix, beta_value, ky, kxValues, S_function, dq_function, du_function, dvt_plus, dvt_minus, first_coefficient)
    (B,L,M,U,Q,u_coefficients,dq_coefficients) = computeMatricesForStabilityAnalysis(sizeOfMatrix,S_function, dq_function, du_function, dvt_plus, dvt_minus, first_coefficient)
    #Should have stable eigenvalues
    (values, vectors) = eig(U, B)
    if(minimum(imag(values))>0)
        println("Stable piece is unstable...")
    end
    
    eigenObjects = []
    for i=1:length(kxValues)
        kx = kxValues[i]
        ul = u_galerkin(sizeOfMatrix, u_coefficients, 1.)+dvt_plus*dot(p_psi_function(sizeOfMatrix, 1.),psi(sizeOfMatrix,kx,ky,M,L,1.))
        ur = -dvt_plus*dot(p_psi_function(sizeOfMatrix, 1.),psi(sizeOfMatrix,kx,ky,M,L,0.))
        ll = dvt_minus*dot(p_psi_function(sizeOfMatrix, 0.),psi(sizeOfMatrix,kx,ky,M,L,1.))
        lr = u_galerkin(sizeOfMatrix, u_coefficients, 0.)-dvt_minus*dot(p_psi_function(sizeOfMatrix, 0.),psi(sizeOfMatrix,kx,ky,M,L,0.))
        lhs = [[ul ur]; [ll lr]]
        eigenResult = eigen(lhs)
        append!(eigenObjects, [eigenResult])
    end
    #println("Done computing eigenvalues and eigenvectors for all kx's")
    return eigenObjects
end

function getCompleteStabilityEigenObjectsOfKx(sizeOfMatrix, beta_value, ky, kxValues, S_function, dq_function, du_function, dvt_plus, dvt_minus, first_coefficient)
    (B,L,M,U,Q,u_coefficients,dq_coefficients) = computeMatricesForStabilityAnalysis(sizeOfMatrix,S_function, dq_function, du_function, dvt_plus, dvt_minus, first_coefficient)
    eigenObjects = []

    for i=1:length(kxValues)
        kx = kxValues[i]
        psi_plus = psi(sizeOfMatrix,kx,ky,M,L,1.)
        psi_minus = psi(sizeOfMatrix,kx,ky,M,L,0.)
        p_psi_plus = p_psi_function(sizeOfMatrix, 1.)
        p_psi_minus = p_psi_function(sizeOfMatrix, 0.)
        centerRowMatrix = ((kx^2+ky^2)*M+L)\B
        
        ul = u_galerkin(sizeOfMatrix, u_coefficients, 1.)+dvt_plus*dot(p_psi_plus,psi_plus)
        uc = -(dvt_plus)*(p_psi_plus'*centerRowMatrix)
        ur = -dvt_plus*dot(p_psi_plus, psi_minus)
        cl = (Q+beta_value*M)*psi_plus
        cc = U-(Q+beta_value*M)*centerRowMatrix
        cr = -(Q+beta_value*M)*psi_minus
        ll = dvt_minus*dot(p_psi_minus,psi_plus)
        lc = -(dvt_minus)*(p_psi_minus'*centerRowMatrix)
        lr = u_galerkin(sizeOfMatrix, u_coefficients, 0.)-dvt_minus*dot(p_psi_minus,psi_minus)
        lhs = [[ul uc ur];[cl cc cr];[ll lc lr]]
        lhs_small = [[ul ur]; [ll lr]]
        
        rhs = [[1 zeros(1,sizeOfMatrix) 0];[zeros(sizeOfMatrix,1) B zeros(sizeOfMatrix,1)]; [0 zeros(1,sizeOfMatrix) 1]]
        eigenResult = eigen(lhs, rhs)
        append!(eigenObjects, [eigenResult])
    end
    #println("Done computing eigenvalues and eigenvectors for all kx's")
    return eigenObjects
end
    
function getStabilityValues(eigenObjectsArray, kxValues)
    steps = length(kxValues)
    growthRates = zeros(steps)
    waveSpeeds = zeros(steps)
    minWaveSpeeds = zeros(steps)
    eigenvectors = []
    
    for i=1:steps
        kx = kxValues[i]
        max_im, max_ind = findmax(imag(eigenObjectsArray[i].values))
        append!(eigenvectors, [eigenObjectsArray[i].vectors[:,max_ind]])
        waveSpeeds[i] = real(eigenObjectsArray[i].values[max_ind])
        minWaveSpeeds[i] = minimum(real(eigenObjectsArray[i].values))
        growthRates[i] = max_im*kx
    end
    
    return growthRates, eigenvectors, waveSpeeds
end

function createFourStabilityPlots(sizeOfMatrix, S_function, growthRates, eigenvectors, waveSpeeds, kx_values, ky_value)
    B = B_ij(sizeOfMatrix)
    L = L_ij(sizeOfMatrix, S_function)
    M = M_ij(sizeOfMatrix)
    
    maxIndex = argmax(growthRates)
    maxKx = kx_values[maxIndex]
    #println("Max growth rate at kx=$maxKx")

    highVector = eigenvectors[maxIndex]
    vectorSize = length(highVector)
    highVector_bottom = highVector[vectorSize] 
    psi_coeffs = highVector[1]*psi(sizeOfMatrix, maxKx, ky_value, M, L, 1.)-highVector_bottom*psi(sizeOfMatrix,maxKx,ky_value, M,L,0.)
    psi_coeffs = psi_coeffs - ( (maxKx^2 + ky_value^2)*M + L ) \ B*highVector[2:end-1]
    
    heights = collect(LinRange(0,1,100))
    psi_angles = zeros(length(heights))
    psi_amps = zeros(length(heights))
    for i=1:length(heights)
        psi_val = galerkin_p_psi_print(sizeOfMatrix, psi_coeffs, heights[i])
        psi_angles[i] = angle(psi_val)
        psi_amps[i] = abs(psi_val)
    end

    #print(kxs[25])
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
