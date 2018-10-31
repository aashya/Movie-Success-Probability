%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('graphdata12.txt');
ticket_price = data(:, 1);
actor_rating= data(:, 2);
tier2pop = data(:, 3);
year = data(:, 4);
season_release = data(:, 5);
theatrestier2 = data(:, 6);
genre = data(:, 7);
awards = data(:, 8);
critics = data(:, 9);
sequel = data(:, 10);
sales = data(:, 11);
X=data(:, [1,2,3,4,5,6,7,8,9,10]);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);
figure
plot(ticket_price, sales);
hold on;

xlabel('Ticket price')
ylabel('Ticket Sales')


figure
plot(ticket_price, sales);
hold on;
plot(tier2pop, sales);
hold on;
plot(theatrestier2, sales);
hold on;

xlabel('Ticket price, population,theatres')
ylabel('Ticket Sales')


figure

ax2 = subplot(4,1,1);
plot(ticket_price, sales);
hold on;
plot(tier2pop, sales);
hold on;
plot(theatrestier2, sales);
hold on;

xlabel('Ticket price, population,theatres')
ylabel('Ticket Sales')


ax2 = subplot(4,3,4);
plot(genre, sales);
hold on;
xlabel('Genre')
ylabel('Ticket Sales')

ax6 = subplot(4,3,7);
plot(actor_rating, sales);
hold on;
xlabel('actor rating')
ylabel('Ticket Sales')

ax4 = subplot(4,3,5);
scatter(awards, sales);
hold on;
xlabel('Awards')
ylabel('Ticket Sales')

ax5 = subplot(4,3,6);
scatter(critics, sales);
hold on;
xlabel('Critic Reviews')
ylabel('Ticket Sales')

ax10 = subplot(4,1,4);
plot(genre, sales);
hold on;
plot(critics, sales);
hold on;
plot(awards, sales);
hold on;
xlabel('Critic Reviews, Genre, Actor Rating')
ylabel('Ticket Sales')
%figure

ax9 = subplot(4,3,9);
scatter(season_release, sales);
hold on;
%plot(critics, sales);
%hold on;
%plot(awards, sales);
%hold on;
%plot(genre, sales);
%hold on;
%hold on;
xlabel('Season')
ylabel('Ticket Sales')

ax7 = subplot(4,3,8);
scatter(year, sales);
hold on;
xlabel('Year of release')
ylabel('Ticket Sales')

% Put some labels 
%hold on;
% Labels and Legend
%xlabel('Factors')
%ylabel('Ticket Sales')

% Specified in plot order

hold off;

%fprintf('\nProgram paused. Press enter to continue.\n');



%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
fprintf('\n its reached compute \n');

[m, n] = size(X);
fprintf('\n size is %f\n', size(X));

% Add intercept term to x and X_test
X = [ones(m, 1) X];

%fprintf('\n X is %f\n', X);

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
fprintf('\n initial theta is %f\n', initial_theta);
fprintf('\n init %f\n', n+1);

fprintf('\n size of initial theta is %f\n', size(initial_theta));


% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, sales);

fprintf('Cost at initial theta (zeros): %f\n', cost);
%fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
%fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

% Compute and display cost and gradient with non-zero theta
test_theta = [159;2;34;2012; 0 ;289;5;3;5;0; 0.5];
[cost, grad] = costFunction(test_theta, X, sales);

fprintf('\nCost at test theta: %f\n', cost);
%fprintf('Expected cost (approx): 0.218\n');
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);
%fprintf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');


%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;
fprintf('\nPART 3 BEGINS \n');

%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
fprintf('\n options done \n');
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, sales)), initial_theta, options);
fprintf('\n theta cost found \n');
% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
%fprintf('Expected cost (approx): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
%fprintf('Expected theta (approx):\n');
%fprintf(' -25.161\n 0.206\n 0.201\n');

% Plot Boundary
%plotDecisionBoundarygraph(theta, X, sales);

%Put some labels 
%hold on;
%Labels and Legend
%xlabel('Exam 1 score')
%ylabel('Exam 2 score')

% Specified in plot order
%legend('Admitted', 'Not admitted')
%hold off;

fprintf('\nPart 4\n');


%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

prob = sigmoid([1 0 2 330 2000 0 233 1 0 0 0] * theta);
fprintf(['For a movie with following -- ticket price -- 0, actor rating -- 2, tier2pop -- 350, year -- 2000, season_release -- no, theatrestier2 -- 233, genre -- drama , awards won -- 2, critics rating -- 5 , sequel -- no, we predict an success rate ' ...
         'probability of %f\n'], prob);
%195 2 46 2005 0 233 3 2 5 1     
%fprintf('Expected value: 0.775 +/- 0.002\n\n');

% Compute accuracy on our training set
p = predict(theta, X);

%fprintf(' p is %f\n', p);
%fprintf(' sales is %f\n', sales);
%fprintf(' d sales is %f\n', mean(double(p == sales)));
%fprintf('Train Accuracy: %f\n', mean(double(p == sales)) * 100);
%fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n');




