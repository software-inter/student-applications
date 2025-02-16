% Визначаємо початкові умови і параметри
R0 = 1000; % початковий розмір червоних сил (партизан)
B0 = R0; % початковий розмір синіх сил (регулярні війська), щоб досягти паритету
beta = 0.05; % параметр знищення для червоних сил
rho = 0.05; % параметр знищення для синіх сил
initial_conditions = [B0; R0]; % початкові умови

% Вирішуємо систему диференційних рівнянь
[t, y] = ode45(@(t, y) guerrilla_warfare(t, y, beta, rho, R0), [0 50], initial_conditions);

% Створюємо перший графік
figure(1)
plot(t, y(:,1), 'b-', 'DisplayName', 'Регулярна армія','LineWidth',2);
hold on;
plot(t, y(:,2), 'r-', 'DisplayName', 'Партизанська армія','LineWidth',2);
xlabel('Час');
ylabel('Розмір армії');
legend;
title('Регулярна армія проти партизанської армії з одинаковими початковими умовами');


% Визначаємо початкові умови і параметри
R0 = 1000; % початковий розмір червоних сил (партизан)
B0 = R0*sqrt(2); % початковий розмір синіх сил (регулярні війська), щоб досягти паритету
beta = 0.5001; % параметр знищення для червоних сил
rho = 0.5; % параметр знищення для синіх сил
initial_conditions = [B0; R0]; % початкові умови

% Вирішуємо систему диференційних рівнянь
[t, y] = ode45(@(t, y) guerrilla_warfare(t, y, beta, rho, R0), [0 15000], initial_conditions);

% Створюємо другий графік
figure(2)
plot(t, y(:,1), 'b-', 'DisplayName', 'Регулярна армія','LineWidth',2);
hold on;
plot(t, y(:,2), 'r-', 'DisplayName', 'Партизанська армія','LineWidth',2);
xlabel('Час');
ylabel('Розмір армії');
legend;
title('Регулярна армія проти партизанської армії з різними початковими умовами');
