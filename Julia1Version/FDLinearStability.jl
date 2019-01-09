using LinearAlgebra

#------------------------------------------------------------------------------
#
# The following functions generate the U, L, and Q matrices for the finite-
# difference linear stability problem
#
#------------------------------------------------------------------------------

function z_elements(sizeN)
    stepSize = 1.0/sizeN
    elements = zeros(sizeN)
    elements[1] = stepSize/2
    for i = 2:sizeN
        elements[i] = elements[i-1]+stepSize;
    end
    return elements
end

function U_fd(sizeN, u_function)
    z_k = z_elements(sizeN)
    for kk=1:sizeN
        z_k[kk] = u_function(z_k[kk])
    end
    fullMatrix = Diagonal(z_k)
    return fullMatrix
end

function L_fd(sizeN, S)
    deltaZ = 1.0/sizeN
    fullMatrix = zeros(sizeN, sizeN)
    fullMatrix[1,1] = S(deltaZ)
    fullMatrix[1,2] = -S(deltaZ)
    fullMatrix[sizeN,sizeN] = S((sizeN-1)*deltaZ)
    fullMatrix[sizeN,sizeN-1] = -S((sizeN-1)*deltaZ)
    for i=2:sizeN-1
        fullMatrix[i, i-1] = -S((i-1)*deltaZ)
        fullMatrix[i,i] = S((i-1)*deltaZ)+S(i*deltaZ)
        fullMatrix[i, i+1] = -S(i*deltaZ)
    end
    normalize = 1.0/(deltaZ^2)
    return fullMatrix*normalize
end

function Q_fd(sizeN, u_function, L_matrix)
    z_k = z_elements(sizeN)
    u_matrix = zeros(sizeN)
    for kk=1:sizeN
        u_matrix[kk] = u_function(z_k[kk])
    end
    diagonals = L_matrix*u_matrix
    fullMatrix = Diagonal(diagonals)
    return fullMatrix
end

#------------------------------------------------------------------------------
#
# The following function solves the finite-difference linear stability problem
#
#------------------------------------------------------------------------------

function growthRateEigenValues_fd(sizeN, beta_value, ky, kxValues, S_function, u_function)
    U = U_fd(sizeN, u_function)
    L = L_fd(sizeN, S_function)
    Q = Q_fd(sizeN, u_function, L)
    #println("FD Matrices Computed")
    
    steps = length(kxValues)
    
    growthRates = zeros(steps)
    waveSpeeds = zeros(steps)
    eigenvectors = []
    for i=1:steps
        #println()
        kx = kxValues[i]
        lhs = U*(Diagonal(zeros(sizeN).+(kx^2+ky^2))+L) - (Q+Diagonal(zeros(sizeN).+beta_value))
        rhs = Diagonal(zeros(sizeN).+(kx^2+ky^2))+L
        #println(lhs)
        eigenObject = eigen(lhs, rhs)
        max_im, max_ind = findmax(imag(eigenObject.values))
        #println(eigenObject[:vectors])
        append!(eigenvectors, [eigenObject.vectors[:,max_ind]])
        waveSpeeds[i] = real(eigenObject.values[max_ind])
        growthRates[i] = max_im*kx
        # If the largest growth rate is zero then there are usually lots of eigenvalues with growth rate 0, and it's hard to pick out which one contains the 'correct' wave speed. Instead we just set the wave speed to NaN since we're mainly interested in growing/unstable modes.
        if growthRates[i] == 0.0
            waveSpeeds[i] = NaN
        end
    end
    return growthRates, eigenvectors, waveSpeeds
end
