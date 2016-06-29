% This script demonstrates the squential linear minimum mean squared error
% in the case of scalar observations
%
% We are trying to estimate the 
% 
% Phillip K Poon
% ppoon@optics.arizona.edu
% 29 Jun 2016
%

%% User Defined Variables



% Length of ground truth signal vector
n = 50

% The number of measurements aka observations
numMeas = 3000

noiseStd = 1

% Initialize the sground truth signal
x = randn(n,1);


% Initialize measurement matrix a. The columns of the vector will be used 
% to form projections with the signal x (aka inner products). The number 
% of columns is equal to the number of measurements. 
a = randn(n,numMeas);

%% Initial several variables

% The matrix where we store the estimated signals
xhat = [];

% Initialize first loop for sequential LMMSE
m = 1

% The first observation aka measurement
y(m) = a(:,m)'*x + noiseStd*randn(1)

% The first estimate is just 0. 
xhat(:,m) = 0*ones(n,1);

% Covariance of the aprior probability density function of x
% We will assume all the entries of x are uncorrelated
Ce = eye(n);

% Initialize the weights
k = [];


% Begin measurement and estimation loop
for m = 2:numMeas
    
    % Simulate the actual measurement
    
    % Noise term
    noiseTerm = noiseStd*randn(1);
    
    % The mth measurements is the inner product of the mth projection
    % vector with the ground truth signal plus the noise
    y(m) = a(:,m)'*x + noiseTerm;
    
    %% LMMSE 
    
    % This section is the code for the sequential linear minimum 
    % mean square error estimator
    
    
    % The new weighting vector
    k(:,m) = ( Ce * a(:,m) ) / ...
        ( noiseStd^2 + a(:,m)'*Ce*a(:,m) );
    
    % Update the covariance matrix
    Ce = (eye(n) - k(:,m) * a(:,m)') * Ce;
    
    % The new estimation
    xhat(:,m) = xhat(:,m-1) + ...
        k(:,m) * ( y(m) - a(:,m)' * xhat(:,m-1) );
    
    % Compute error
    mmseE = xhat(:,m) - x;
    
    % 
    mmseErrorList(m) = (1/n)*trace( mmseE*mmseE' );
    
    
       
    figure(100);
    subplot(3,1,1)
    plot(x)
    title('Ground Truth Signal')
    subplot(3,1,2)
    plot(xhat(:,m))
    title('LMMSE Estimated Signal')
    subplot(3,1,3)
    plot(mmseE)
    ylim([-2 2])
    title('Diff. Between Estimate and Truth')

    
    %% LSE
    
    % This code is for the Least Squares Estimator. Although it is in 
    % matrix form and not in sequential form I assume they should give
    % the same answer. 
    
    H(m,:) = a(:,m)';
    %g(m) = h(m,:)*x + noiseTerm;
    g = y';
    
    
    xhat_ls = (pinv(H'*H)*H'*g);
    
          
      
    % Compute error aka the difference between the estimated signal x
    % and the ground truth signal x
    lsE = xhat_ls - x;
    
    lsErrorList(m) = (1/n)*trace( lsE*lsE' );
    
    
    
    figure(101);
    subplot(3,1,1)
    plot(x)
    subplot(3,1,2)
    plot(xhat_ls)
    subplot(3,1,3)
    plot(lsE)
    ylim([-2 2])
    
    % Plots the mean squared error    
    figure(105);
    semilogy(mmseErrorList)
    title('Ground Truth Signal')
    hold all
    semilogy(lsErrorList)
    title('LSE Estimated Signal')
    hold off
    xlim([0 numMeas]);
    title('Diff. Between Estimate and Truth')
    grid on
    legend('LMMSE','LSE')
    drawnow
    pause(0.2);

end
