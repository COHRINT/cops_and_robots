%salmondjoingmm.m
%gmmout = salmondjoingmm(gmmin,nmixandsout)
%Merges together components from a gaussian mixture model object (gmmin)
%using Salmond's joining algorithm, such that the output contains
%nmixandsout components with the same mixture mean and mixture covariance
%as gmmin.
%***NOTE!: this code is not the most efficient implementation of Salmond's 
%joining algorithm, as the weighted distance matrix between mixands DIJ is
%recomputed at every iteration. A smarter and faster implementation of this 
%algorithm would be to NOT recompute DIJ at every iteration, but rather to 
%preserve the parts of DIJ that do not get merged after each iteration and 
%to only compute a new row/col of DIJ for the newly merged component 
%on each iteration. 
function gmmout = salmondjoingmm(gmmin,nmixandsout)
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

%%compute input mixture's mean and covariance
mumixin = sum(muin'.*repmat(pin,[dim,1]),2); % mean (col vector)
muin_minus_mumixin = muin' - repmat(mumixin,[1,ncomps]);
Pmixin = zeros(dim,dim); %covariance
for jj=1:ncomps
    Pmixin = Pmixin + pin(jj)*(Sigmain(:,:,jj) +...
             muin_minus_mumixin(:,jj)*muin_minus_mumixin(:,jj)');
end
invPmixin = inv(Pmixin);
ncompsout = ncomps;
ptemp = pin;
mutemp = muin';
Sigmatemp = Sigmain;

while(ncompsout>nmixandsout)
   %%compute product and sum of all weights
   [wtgrid1,wtgrid2] = meshgrid(ptemp);
   prodwts = wtgrid1.*wtgrid2;
   sumwts = wtgrid1 + wtgrid2;
   %%compute difference between all means
   diffallmeans = repmat(mutemp,[1,ncompsout])-rcecho(mutemp,2,ncompsout);
   quadproducts = reshape(sum(diffallmeans'*invPmixin.*diffallmeans',2),[ncompsout,ncompsout]);
   DIJ = (prodwts)./(sumwts) .* quadproducts;
   %%set diagonal entries to Inf
   Infmask = diag(Inf*ones(1,ncompsout)).*eye(ncompsout);
   DIJ = DIJ+Infmask;
   %%Minimize DIJ
   [dvalmin,minind] = min(DIJ(:)); %find first index for min
   [mi,mj] = ind2sub(ncompsout,minind); %translate to subscript indices
   %%keep track of non-merged components
   nonmergedinds = 1:ncompsout;
   nonmergedinds = nonmergedinds(nonmergedinds~=mi & nonmergedinds~=mj);
   %%merge components i,j:
   pmerged = ptemp(mi) + ptemp(mj);
   mui = mutemp(:,mi); muj = mutemp(:,mj);
   mumerged = (ptemp(mi)*mui + ptemp(mj)*muj)/pmerged;
   Sigmamerged = (ptemp(mi)*Sigmatemp(:,:,mi) +ptemp(mj)*Sigmatemp(:,:,mj)...
                 + (prodwts(minind)./(pmerged))*(mui-muj)*(mui-muj)')...
                 /pmerged;
   %%update stats for gmmout
   ncompsout = ncompsout-1; %update number of components now in output    
   ptemp = [ptemp(nonmergedinds),pmerged];
   mutemp =[mutemp(:,nonmergedinds),mumerged];
   Sigmatemp = cat(3,Sigmatemp(:,:,nonmergedinds),Sigmamerged);
end
gmmout = gmdistribution(mutemp',Sigmatemp,ptemp);