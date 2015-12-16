function [ w,b ] = svm_solver(x,y,C)
x = cat(2, x, y);
l = size(x,1);
f = -1*ones(l,1);
H = zeros(l);
for i = 1:l
    for j = 1:l
        xi = x(i,1:end-1)'; xj = x(j,1:end-1)';
        H(i,j) = x(i,end) * x(j,end) * (xi'*xj);
    end
end
% load('H.mat', 'H');

Aeq = x(:,end)'; %y_i
Aeq = double(Aeq);
ep = 10^(-5);
beq = 0;
lb = ep*ones(l,1) ; ub = (C - ep) * ones(l,1);
alp = quadprog(H,f,[],[],Aeq,beq,lb, ub, [], optimset('Algorithm','interior-point-convex','Display','off'));

w = sum(repmat(alp.*x(:,end),[1 size(x,2)-1]).*x(:,1:end-1));
b = x(:,end)' - w*x(:,1:end-1)';
b = mean(b);

end