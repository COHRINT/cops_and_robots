%test_GMmerging.m
clc,clear
close all
rng(100)
[X,Y] = meshgrid(-20:0.1:20);
XY = [X(:) Y(:)];

ncA = 40;
muA = -12 + 24*rand(ncA,2);
crosscovars = -0.5 + rand(1,ncA);
SigA = repmat(eye(2),[1 1 ncA]) + reshape([zeros(1,ncA);...
                                         crosscovars;...
                                         crosscovars;...
                                         zeros(1,ncA)],[2 2 ncA]);
%%add random noise to the diagonals
SigA = SigA + reshape([2.5*rand(1,ncA);zeros(2,ncA);2.5*rand(1,ncA)],[2 2 ncA]);                                     
% pwtsA = rand(1,ncA); pwtsA = pwtsA./sum(pwtsA);
pwtsA = dirichletrnd(ones(1,ncA));
gmA = gmdistribution(muA,SigA,pwtsA);
pdfA = reshape(pdf(gmA,XY),size(X));
figure(),set(gcf,'Position',[70   540   560   420])
surf(X,Y,pdfA,'EdgeColor','none'),view(2)
title('pdf A')
colormap('hot')

nmixandsout = 5;

% %%Merge with Salmond
% tic %tstart = cputime;
% gmmoutS = salmondjoingmm(gmA,nmixandsout);
% toc %tfin = cputime - tstart

%%Merge with Runnalls
tic %tstart = cputime;
gmmoutR = runnallsjoingmm(gmA,nmixandsout);
toc %tfin = cputime - tstart

% %%Merge with KLD k-means
% tic %tstart = cputime;
% gmmoutK = kmeansKLDjoingmm(gmA,nmixandsout,10);
% toc %tfin = cputime - tstart

%%

% pdfS = reshape(pdf(gmmoutS,XY),size(X));
% figure()
% surf(X,Y,pdfS,'EdgeColor','none'),view(2)
% title('Salmond Merged GM')
% colormap('hot')

pdfR = reshape(pdf(gmmoutR,XY),size(X));
figure()
surf(X,Y,pdfR,'EdgeColor','none'),view(2)
title('Runnalls Merged GM')
colormap('hot')

% pdfK = reshape(pdf(gmmoutK,XY),size(X));
% figure()
% surf(X,Y,pdfK,'EdgeColor','none'),view(2)
% title('k-means KLD Merged GM')
% colormap('hot')