%runallsjoingmm.m
%gmmout = runallsjoingmm(gmmin,nmixandsout)
%Merges together components from a gaussian mixture model object (gmmin)
%using Runnalls' joining algorithm, such that the output contains
%nmixandsout components that are formed by moment merging to minimize an 
%upper bound to the KL divergence to the original mixture.
%***NOTE!: this code is not the most efficient implementation of Runnalls' 
%joining algorithm, as the weighted Bij matrix between mixands BIJ is
%recomputed at every iteration. A smarter and faster implementation of this 
%algorithm would be to NOT recompute BIJ at every iteration, but rather to 
%preserve the parts of BIJ that do not get merged after each iteration and 
%to only compute a new row/col of BIJ for the newly merged component 
%on each iteration.
function gmmout = runnallsjoingmm(gmmin,nmixandsout)
ncomps = gmmin.NComponents;
if ncomps<nmixandsout
   gmmout = gmmin;
   %warning('Number of mixands less than max number of mixands')
   return %leave without doing anything
end
pin = gmmin.PComponents;
muin = gmmin.mu;
Sigmain = gmmin.Sigma;
%dim = size(muin,2); %dimension of GM components

mutemp = muin';
Sigmatemp = Sigmain;
ptemp = pin;
ncompsout = ncomps;
while(ncompsout>nmixandsout)
%    %%compute product and sum of all weights
%    [wtgrid1,wtgrid2] = meshgrid(ptemp);
%    prodwts = wtgrid1.*wtgrid2;
%    sumwts = wtgrid1 + wtgrid2;
   %%compute difference between all means
%    diffallmeans = repmat(mutemp,[1,ncompsout])-rcecho(mutemp,2,ncompsout);

   %%compute BIJ matrix for first time if needed
   if ~exist('BIJ','var')
       BIJ = inf(ncompsout);
       for ii=1:ncompsout
           wi = ptemp(ii);
           mui = mutemp(:,ii);
           Pi = Sigmatemp(:,:,ii);
           logdetPi = log(det(Pi));
           for jj=1:ncompsout
               if ii<jj
                   wj = ptemp(jj);
                   muj = mutemp(:,jj);
                   Pj = Sigmatemp(:,:,jj);
                   logdetPj = log(det(Pj));
                   wijsum = wi + wj;
                   Pij = (wi/wijsum)*Pi + (wj/wijsum)*Pj + ...
                       (wi/wijsum)*(wj/wijsum)*(mui-muj)*(mui-muj)';
                   BIJ(ii,jj) = 0.5*( wijsum*log(det(Pij)) - wi*logdetPi ...
                       - wj*logdetPj);
                   BIJ(jj,ii) = BIJ(ii,jj); %exploit symmetry            
               elseif ii>=jj
                   continue
               end
           end
       end       
   end
      
   %%Minimize BIJ
   [dvalmin,minind] = min(BIJ(:)); %find first index for min
   [mi,mj] = ind2sub(ncompsout,minind); %translate to subscript indices
   %%keep track of non-merged components
   nonmergedinds = 1:ncompsout;
   nonmergedinds = nonmergedinds(nonmergedinds~=mi & nonmergedinds~=mj);
   %%merge components i,j:
   pmerged = ptemp(mi) + ptemp(mj);
   mui = mutemp(:,mi); muj = mutemp(:,mj);
   mumerged = (ptemp(mi)*mui + ptemp(mj)*muj)/pmerged;
   Sigmamerged = (ptemp(mi)*Sigmatemp(:,:,mi) +ptemp(mj)*Sigmatemp(:,:,mj)...
                 + (ptemp(mi)*ptemp(mj)./(pmerged))*(mui-muj)*(mui-muj)')...
                 /pmerged;
   logdetPmerged = log(det(Sigmamerged));          
   %%update stats for gmmout
   ncompsout = ncompsout-1; %update number of components now in output    
   ptemp = [ptemp(nonmergedinds),pmerged];
   mutemp =[mutemp(:,nonmergedinds),mumerged];
   Sigmatemp = cat(3,Sigmatemp(:,:,nonmergedinds),Sigmamerged);
   %% Update existing version of BIJ
       %%-->set entries for merged elements to NaN
       BIJ(mi,:) = NaN;
       BIJ(mj,:) = NaN;
       BIJ(:,mi) = NaN;
       BIJ(:,mj) = NaN;
       %%-->select only the relevant elements from last iteration
       BIJ = reshape(BIJ(~isnan(BIJ)),[ncompsout-1,ncompsout-1]); %note: at this point BIJ has just lost 2 rows and 2 cols                                                    
       BIJ = [BIJ, inf(ncompsout-1,1);
              inf(1,ncompsout-1), inf]; %augment for new merger component
       for ii = 1:ncompsout-1
          wi = ptemp(ii);
          mui = mutemp(:,ii);
          Pi = Sigmatemp(:,:,ii);
          logdetPi = log(det(Pi));
          wsum = pmerged + wi;
          
          Pimerged = (wi/wsum)*Pi + (pmerged/wsum)*Sigmamerged + ...
                       (wi/wsum)*(pmerged/wsum)*(mui-mumerged)*(mui-mumerged)';
          BIJ(ii,end) = 0.5*( wsum*log(det(Pimerged)) ...
                              - wi*logdetPi - pmerged*logdetPmerged);
          BIJ(end,ii) = BIJ(ii,end);
       end
end
gmmout = gmdistribution(mutemp',Sigmatemp,ptemp);