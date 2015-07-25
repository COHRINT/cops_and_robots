function gmmout = kmeanjoingmm(gmmin,nmixandsout)
ncomps = gmmin.NComponents;
if ncomps<nmixandsout
   gmmout = gmmin;
   %warning('Number of mixands less than max number of mixands')
   return %leave without doing anything
end
pin = gmmin.PComponents;
muin = gmmin.mu;
Sigmain = gmmin.Sigma;
dim = size(muin,2); %dimension of GM components

mutemp = muin';
Sigmatemp = Sigmain;
ptemp = pin;
ncompsout = ncomps;

muout = zeros(dim,nmixandsout);
Sigmaout = zeros(dim,dim,nmixandsout);
pout = zeros(1,nmixandsout);

ind = kmeans(muin, nmixandsout,'EmptyAction','singleton','Replicates',1);

for clustNo = 1:nmixandsout
    clustInd = find(ind == clustNo);
   
    %%merge all components in cluster using moment matching
    pwtsi = ptemp(clustInd);
    pmerged = sum(pwtsi);
    mui = mutemp(:,clustInd);
    mumerged = sum(repmat(ptemp(clustInd),[dim, 1]).*mui, 2)/pmerged;
    Sigmai = Sigmatemp(:,:,clustInd);
    Sigmasum = 0;
    for i = 1:length(clustInd)
        Sigmasum = Sigmasum + (pwtsi(i)/pmerged)*(Sigmai(:,:,i) ...
                                                           + mui(:,i)*mui(:,i)');
    end
%     Sigmamerged = (Sigmasum ...
%                  + (prod(ptemp(clustInd))./(pmerged))*diff(mui)*diff(mui)')...
%                  /pmerged;
    Sigmamerged = Sigmasum - mumerged*mumerged'; 

    %%save for output
	muout(:,clustNo) = mumerged;
    Sigmaout(:,:,clustNo) = Sigmamerged;
    pout(clustNo) = pmerged;
end

gmmout = gmdistribution(muout',Sigmaout,pout);             
             