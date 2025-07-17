% Parameters
% xo: odd data (1:2:end)
% xe: even data (2:2:end - 1)
% yo: odd recon, after C (conv) and P (pull) have been applied
% ye: even recon, after C (conv) and P (pull) have been applied
% s: scaling parameter
% t: noise precision
syms xo xe yo ye s real
syms t positive

% Energy function (E = t/2 ||x - Sy||_2^2)
% where x = [xo xe]', y = [yo ye]', S = [exp(s) 0; 0 exp(-s)]
E = 0.5*t*((xo - exp(s)*yo)^2 + (xe - exp(-s)*ye)^2);

% Gradient
g_s = diff(E, s);
g_s = simplify(g_s, 1000)
% >> t*ye*exp(-s)*(xe - ye*exp(-s)) - t*yo*exp(s)*(xo - yo*exp(s))

% Hessian
H_s = diff(g_s, s);
% Expectation of the Hessian (Fisher scoring)
HH_s = simplify(subs(H_s, xo, exp(s)*yo), 1000);
HH_s = simplify(subs(HH_s, xe, exp(-s)*ye), 1000)
% >> t*exp(-2*s)*(exp(4*s)*yo^2 + ye^2)
