
load twotankdata
z = iddata(y, u, 0.2, 'Name', 'Two tank system');

z1 = z(1:1000);
z2 = z(1001:2000);
z3 = z(2001:3000);
plot(z1,z2,z3)
legend('Estimation','Validation 1', 'Validation 2')


V = arxstruc(z1,z2,struc(1:5, 1:5,1:5));
% select best order by Akaike's information criterion (AIC)
% This is something that i should change later on
nn = selstruc(V,'aic');

mw1 = nlarx(z1,[5 1 3], wavenet);

NLFcn = mw1.OutputFcn;
% NLFcn.NonlinearFcn.NumberOfUnits

% compare(z1,mw1);


