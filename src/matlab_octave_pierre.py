from oct2py import octave
octave.addpath('home/behollis/winter2013/pierre_lock_exch/DOorthnorm.m')
octave.addpath('home/behollis/winter2013/pierre_lock_exch/PlotScript_special.m')


'''
Hi Brad,

Thanks for the email. I hope you had a good summer internship.

We have prepared a new data set for you. It is based on our new methodology and numerical algorithm for uncertainty prediction, see these papers:
o       http://mseas.mit.edu/publications/PDF/ueckermann_etal_DO-numerics_JCP2011.pdf
o       http://mseas.mit.edu/publications/PDF/sapsis_lermusiaux_DO_SDE_PhysD2009.pdf
You should likely read the start of the first JCP paper as well as the appendix A in that paper, but you don't need to read both of these papers in detail.

Matt Ueckermann in our group (cc-ed) defined and ran the specific example that we give you. It is similar to the examples described in Sec 6.1 and 6.2 of the first JCP paper mentioned above (Matt is the first author of that paper). It is a lock-exchange problem: the initial conditions are heavy fluid on one side and light fluid on the other, separated by a barrier (the lock). At initial time, that barrier is removed, and the flow is allowed to evolve. Initial uncertainties in the example we give you are different than what we published in that first JCP paper above: it now originates from not knowing the position of the interface between the two fluids. In other words, the volumes of heavy and light fluid on each side is not exactly known, and the initial barrier slides left and right accordingly. Initially, the initial probability distribution of the position of the barrier is Gaussian. Therefore, after an infinite time, I would expect a similar Gaussian distribution, but with the light fluid on top of the heavy one, and with the variance of distribution possibly stretched if the size of the whole lock domain is not square. However, the probability distributions of the interface or the dominant dynamics in between this start and infinite time are not necessarily Gaussian.

We give you two files. You can grab them from:
http://mseas.mit.edu/download/pierrel/alex_brad/

-------File 1:        all_DO_runs_mov_snapshots.zip

In that file, you have snapshots and time-dependent movies. It is in part to help you make sure you can read the data files properly. The snapshots and time-dependent movies consist of:

mean   |   evolution of energy/variance of each mode   | realization 1  (= mean + Sum of modes x coefficients_of_realization1)
_________________________________________________
mode 1 |                       marginal pdf of mode 1                   | realization 2 (= mean + Sum of modes x coefficients_of_realization1)
_____________________________________________
mode 2 |                       marginal pdf of mode 2                   | realization 3 (= mean + Sum of modes x coefficients_of_realization1)
   ...                                            ...                        ...

You have snapshots from the start, every 20*50 time steps, up to 201*50 time-steps. We give you plots for two cases: one where we use 15 modes in total and one where we use 20 modes. For the plots of fields (mean and modes), we show velocity streamlines overlaid on the density.

The realizations are sample path evolution of the flow: I believe that there are 1000 of them in what we give you.

I believe the parameters of the runs are (Matt may need to confirm and give a bit more info):
Nx: 128
Ny: 128
T: 15  (non-dimensional)
dt: 4.0000e-04
nu: 2.5000e-04
kappa: 2.5000e-04
S: 15 or 20  (total number of modes)
PlotIntrvl: 50  (number of time-step/interval between plot times)
var: 0.0500
MC: 1000
PrRa: 1

-------File 2:                DO_lock_exchange.tar.gz

This is the file that contains all of the data. The file contains two directories, one for the run with 15 modes (S15) and one for the run with 20 modes (S20). All data files are matlab files.

We also plan to email you a quick matlab script that reads these data files and plots them. I think this will help you get started.

---- Of course, we have several other examples that we can give you (lid-driven cavity, flow over cylinders, etc).

Finally, I have not forgotten that I also want to give you the total velocities for the Mass Bay ensemble simulations that I had given Alex several years ago. I simply have not had to the time to compute them. However, we have not forgotten.

Once you have downloaded the files and the papers, let us know. Then, we can remove them from the site to save space.

All the best,

Pierre

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hello Brad, Alex,

Pierre asked me to make a MATLAB plot script to illustrate how to access
different parts of the solution (mean/modes/stochastic coefficient,
velocity/density) and also how to reconstruct a realization. I've attached this
script along with a helper script. Let me know if you have trouble with it.

I also did a quick check to see what the non-dimensional parameters are for this
simulation. Here we have:

Gr = 1.6e7
Sc = 1

[Also, note that the "PrRa: 1" in the message below is a place-holder variable
for when we work with temperature instead of directly with density, so you may
ignore it.]

Regards,
Matt

'''

''''
example usage from: http://pythonhosted.org/oct2py/examples.html

roundtrip.m

function [x, class] = roundtrip(y)
   % returns the input variable and its class
   x = y
   class = class(x)

Python Session

>>> from oct2py import octave
>>> import numpy as np
>>> x = np.array([[1, 2], [3, 4]], dtype=float)
>>> out, oclass = octave.roundtrip(x)
>>> # or octave.call('roundtrip', x)
>>> # or octave.call('roundtrip.m', x)
>>> # or octave.call('/path/to/roundtrip.m', x)
>>> import pprint
>>> pprint.pprint([x, x.dtype, out, oclass, out.dtype])
[array([[ 1.,  2.],
       [ 3.,  4.]]),
 dtype('float64'),
 array([[ 1.,  2.],
       [ 3.,  4.]]),
 'double',
 dtype('float64')]
'''


'''
%DOorthnorm.m
function [YYt ui vi Pi rhoi] = DOorthnorm(app, dx, dy, YYt, ui, uid, vi, vid, Pi, pid, rhoi)
% function [YYt u v P rho a] = DOorthnorm(YYt, ui, uid, vi, vid, Pi, pid, rhoi)
% This function orthonormalizes the DO modes while preserving the original
% realizations (upto a correction for the stochastic energy) and the
% stochastic energy.
%
% INPUTS:
%   dx:         Size of x discretization
%   dy:         Size of y discretization
%   YYt(MC,S):  Stochastic coefficients
%   ui(:,S):    Mode shapes for u-velocity
%   uid:        Vector of ids for interior u-velocities
%   vi(:,S):    Mode shapes for v-velocity
%   vid:        Vector of ids for interior v-velocities
%   Pi(:,S):    Mode shapes for Pressure
%   Pid:        Vector of ids for interior Pressures
%   rhoi(Optional): Mode shapes for density
%
% OUTPUTS:
%   YYt(MC,S):  Stochastic coefficients
%   ui(:,S):    Mode shapes for u-velocity
%   vi(:,S):    Mode shapes for v-velocity
%   Pi(:,S):    Mode shapes for Pressure
%   rhoi(Optional): Mode shapes for density
%
% Written by: Matt Ueckermann

if ~isfield(app,'IP'),app.IP=[1,1,1];end
if ~isfield(app,'Sb'), app.Sb = 0;end

%% Old orthonormalization procedure
if isfield(app,'GSorth')
%     display('Using old orthonormalization');
    for i=1:app.S(1)
        for j=1:app.S(1)
            if i~=j
                if nargin == 11
                    coef =app.IP(1) * InProd(ui(:,i), ui(:,j), dx, dy, uid)...
                        + app.IP(2) * InProd(vi(:,i), vi(:,j), dx ,dy, vid)...
                        + app.IP(3) * InProd(rhoi(:,i), rhoi(:,j), dx, dy, pid);
                    rhoi(:,i) = rhoi(:,i) - coef * rhoi(:,j);
                else
                    coef = InProd(ui(:,i), ui(:,j), dx, dy, uid)...
                        + InProd(vi(:,i), vi(:,j), dx, dy, vid);
                end
                ui(:,i) = ui(:,i) - coef * ui(:,j);
                vi(:,i) = vi(:,i) - coef * vi(:,j);
            end
        end
        if nargin == 11
            nrm = sqrt(app.IP(1)*InProd(ui(:,i), ui(:,i), dx, dy, uid)...
                + app.IP(2)*InProd(vi(:,i), vi(:,i), dx ,dy, vid)...
                + app.IP(3)*InProd(rhoi(:,i), rhoi(:,i), dx, dy, pid));
            rhoi(:,i) = rhoi(:,i) / nrm;
        else
            nrm = sqrt(InProd(ui(:,i), ui(:,i), dx, dy, uid)...
                + InProd(vi(:,i), vi(:,i), dx, dy, vid));
        end        
        ui(:,i) = ui(:,i) / nrm;
        vi(:,i) = vi(:,i) / nrm;
    end
else
    %% First rotate the stochastic coefficients so that we have
    %% uncorrelated samples    
    Sintid = app.Sb+1:app.S;
    CYY=cov(YYt(:, Sintid));
    [VC, tmp] = eig(CYY) ; %Diagonal Covariance, and Diagonal Vectors
    ui(:, Sintid) = fliplr(ui(:, Sintid) * VC);
    vi(:, Sintid) = fliplr(vi(:, Sintid) * VC);
    Pi(:, Sintid) = fliplr(Pi(:, Sintid) * VC);
    if nargin == 11
        rhoi(:,Sintid) = fliplr(rhoi(:, Sintid) * VC);
    else
        rhoi = 0;
    end
    YYt(:,Sintid) = fliplr(YYt(:,Sintid) * VC);
    
    %% Next orthonormalize the modes
    if nargin == 11
        M = (app.IP(1)*ui(uid, :)' * ui(uid, :) + app.IP(2)*vi(vid, :)' * vi(vid, :) ...
            + app.IP(3) * rhoi(pid, :)' * rhoi(pid, :)) * dx * dy;
    else
        M = (ui(uid, :)' * ui(uid, :) + vi(vid, :)' * vi(vid, :)) * dx * dy;
    end
    [VC DC] = eig(M);
    YYt = fliplr(YYt * VC * sqrt(DC));
%     YYt = fliplr(YYt * VC);
    DC = diag (1 ./ diag(sqrt(DC)));
    ui = fliplr(ui * VC * DC);
    vi = fliplr(vi * VC * DC);
    if nargin == 11
        rhoi = fliplr(rhoi * VC * DC);
    end
    Pi = fliplr(Pi * VC * DC);
    Nbcs = length(ui) - length(uid);
    %Now rotate the modes such that the boundary is orthogonal once again
    %Now do the boundary-inner product to separate edge modes from non-boundary
    %modes
    if nargin == 11
        [VC,DC] = eig(ui(1:Nbcs,:)'*ui(1:Nbcs,:) + vi(1:Nbcs,:)'*vi(1:Nbcs,:) + ...
            rhoi(1:Nbcs,:)'*rhoi(1:Nbcs,:));
        rhoi = fliplr(rhoi*(VC));
    else
         [VC,DC] = eig(ui(1:Nbcs,:)'*ui(1:Nbcs,:) + vi(1:Nbcs,:)'*vi(1:Nbcs,:));
    end
    ui = fliplr(ui*(VC));
    vi = fliplr(vi*(VC));
    Pi = fliplr(Pi*(VC));
    YYt = fliplr(YYt*(VC));

    %% Rotate the stochastic coefficients back to the uncorrelated case
    CYY=cov(YYt(:, Sintid));
    [VC,DC] = eig(CYY) ; %Diagonal Covariance, and Diagonal Vectors
    ui(:, Sintid) = fliplr(ui(:, Sintid) * VC);
    vi(:, Sintid) = fliplr(vi(:, Sintid) * VC);
    Pi(:, Sintid) = fliplr(Pi(:, Sintid) * VC);
    if nargin == 11
        rhoi(:, Sintid) = fliplr(rhoi(:, Sintid) * VC);
    end
    %Also correct for, or make sure that, stochastic energy is preseved.
    YYt(:, Sintid) = fliplr(YYt(:, Sintid) * VC * ...
        sqrt(sum(tmp(:)) / sum(DC(:))));
%     YYt=fliplr(YYt * VC);
end
'''
'''
%PlotScript_special.m
clear all, clc, clf ,close all

%This is the root directory where the saved files are stored
path = '../../Save/UCSC_vis/S15';

%load param file
load(sprintf('%s/param.mat', path));

%Load the desire output file
sol = load(sprintf('%s/00050.mat', path));

%Orthonormalize the solution (to plot solution in a consistent rotation of
%the coordinate system). 
[sol.YYt sol.ui sol.vi sol.Pi, sol.rhoi] = ...
        DOorthnorm(app, dx, dy, sol.YYt, sol.ui, uid, sol.vi, vid,...
        sol.Pi, pid, sol.rhoi);
    
    
%Plot the mean density, mean u-velocity, and mean v-velocity
figure(1)
pcolor(XP, YP, sol.rho(NodeP(2:app.Ny+1, 2:app.Nx+1))), shading interp
figure(2)
pcolor(XU, YU, sol.u(Nodeu(2:app.Ny+1, 2:app.Nx))), shading interp
figure(3)
pcolor(XV, YV, sol.v(Nodev(2:app.Ny, 2:app.Nx+1))), shading interp

%Plot one of the modes for density, u-velocity, and v-velocity
%Which mode to plot
modeno = 3;
%Plots
figure(4)
rhoi = sol.rhoi(:, modeno);
ui = sol.ui(:, modeno);
vi = sol.vi(:, modeno);
pcolor(XP, YP, rhoi(NodeP(2:app.Ny+1, 2:app.Nx+1))), shading interp
figure(5)
pcolor(XU, YU, ui(Nodeu(2:app.Ny+1, 2:app.Nx))), shading interp
figure(6)
pcolor(XV, YV, vi(Nodev(2:app.Ny, 2:app.Nx+1))), shading interp
%plot the 1D marginal of the same mode
figure(7)
ksdensity(sol.YYt(:, modeno))

%Now, create a realization, and plot the density, u-velocity and v-velocity
%of that realization
realno = 500;
rho_r = sol.rho + sol.rhoi * YYt(realno,:)';
u_r = sol.u + sol.ui * YYt(realno,:)';
v_r = sol.v + sol.vi * YYt(realno,:)';
%Plots
figure(8)
pcolor(XP, YP, rho_r(NodeP(2:app.Ny+1, 2:app.Nx+1))), shading interp
figure(9)
pcolor(XU, YU, u_r(Nodeu(2:app.Ny+1, 2:app.Nx))), shading interp
figure(10)
pcolor(XV, YV, v_r(Nodev(2:app.Ny, 2:app.Nx+1))), shading interp
'''