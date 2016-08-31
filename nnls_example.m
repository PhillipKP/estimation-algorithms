% This is a no-thrills implementation of the non-negative least squares
% algorithm with lots of comments to explain what's going on.
%
% Credit goes to the paper titled "Nonnegativity constraints in numerical
% analysis" by Chen and Plemmons and the paper "A fast
% non-negativity-constrained least squares algorithm" by Bro and De Jong.
%
% The non-negative least squares algorithm is an active set algorithm.

% The number of rows in the matrix A
m = 2
% The number of columns in the matrix A. 
n = 2
% There are n inequality constraints. The nth constrain is said to be
% active, if the nth regression coefficient will be negative (or zero) if
% estimated unconstrained, otherwise the constraint is passive.

% tolerance for the stopping criterion
t = 10^-6;

A = rand(m,n);

xTrue = rand(n,1)

xTrue(1) = -0.2
xTrue(2) = -0.1

y = A*xTrue;


%save('dataset1.mat','A','xTrue','y')


%%

% Matlab's built-in non-negative least squares function for comparison
xMatlab = lsqnonneg(A,y);


%%

%Define the function for squared error
se = @(x,y,A)( (y-A*x).'*(y-A*x) )



% Initialize guess solution
x = zeros(n,1)


% Initialize the passive set P
P = [];

% Initialize the active set R
R = 1:n;

% The actual gradient of the squared error is 2A.'*(Ax -x)
% w = -(1/2) gradient of the squared error.
% So positive elements of w mean a negative slope.
% While negative elements of w mean a positive slope.
w = A.'*(y-A*x)

% The max of w is the component most negative slope.
[maxW, j] = max(w);

count1 = 1;


while (length(R) > 0)  && maxW > t
    
    % The variable associated largest negative slope
    % j is the index associated with the component of largest change
    % (maximum of W), concatenate j to P
    P = [P j]
    
    % Remove j from R
    R = R( R ~= j )
    
    % Let Ap be A restricted to the variables included in the passive set P
    Ap = A(:,P);
    
    % reset s to be all zeros
    s = zeros(n,1);
    
    
    % Computes the unconstrained solution 
    
    % Compute the entries of s that are related to the indices store in the
    % P. Let sp be the psuedoinverse of Ap with y
    sp = pinv( Ap.' * Ap) * (Ap).' * y;
    
    s(P) = sp;
    
    % When a new variable has been included in the passive set P, there is
    % a chance that in the unconstained solution to the new least squares
    % problem sp, tge sine if tge regression coefficients will turn
    % negative. We call the new estimate s and the old estimate x.
  
    % The squared error of the old solution
    se(x,y,A)
    % The squares error of the new solution
    se(s,y,A) % This should always be lower then the value from the previous line!
    
    while min(sp) <= 0
        
        % If we are in this while loop it means the new estimate s has a
        % negative entry in it.
        %
        % However the old estimate x doesn't have any negative entries in
        % it but it's squared error is higher. 
        %
        % Somewhere along the line segment x + alpha(s-x) the 
       
        
        xList = [];
        for i = P
            if s(i) <= 0
                tempVar =  x(i) /  ( x(i) - s(i) );
                xList = [xList tempVar];
            end
        end
        alpha = min( xList );
        
        x = x + alpha*(s - x);
        
        % Move the R all indices j in P such at all x(j) = 0
        
        for j = P
            if x(j) == 0
                
                % Remove j from P
                P = P(P~=j);
                
                % Add j to R
                R = [R j];
            end
        end
        
        % Let Ap be A restricted to the variables included in P
        Ap = A(:,P);
        
        % Compute the entries of s that are related to the indices store in the
        % P. Let sp be the psuedoinverse of Ap with y
        sp = pinv( Ap.' * Ap) * (Ap).' * y;

        % reset s to be all zeros
        s = zeros(n,1);
        
        s(P) = sp;

    end
    
    % rebuild s
    x = s;
    
    % Recompute gradient
    w = A'*(y-A*x)
    
    
    
    [maxW, j] = max(w)
    
    P
    R
    x
    count1 = count1 + 1;
    
end

(y-A*xTrue).'*(y-A*xTrue)
(y-A*x).'*(y-A*x)
xTrue
x
xMatlab