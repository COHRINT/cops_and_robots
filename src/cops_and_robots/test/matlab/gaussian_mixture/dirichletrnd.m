%dirichletrnd.m
%Draws random sample from Dirichlet distribution over k-ary multinomial pdf using
%Gamma pdf sampling pdf procedure given Dirichlet parameter vector alpha =
%[alpha1,alpha2,...,alphak], where alphai is a real number.
%(1) for i=1,...,k, generate zi ~ Gamma(alphai,1)
%(2) normalize to get qi: form qi = zi/sum_{i=1}^{k}{zi}
%(3) return Dirichlet sample as q = [q1,q2,...,qk].

function qout = dirichletrnd(alphavec)
%%Generate zi
z = gamrnd(alphavec,1);
sumz = sum(z);
%Normalize and return if not all elements are zero
qout = z/sumz;
if sumz<realmin   
    %pick a random element to be 1 and set all others to zero
    %(? I think this is equivalent to "conditioning" this term on all 
    %others being equal to zero)
    rind = randperm(length(z));
    qout = zeros(size(z));
    qout(rind(1)) = 1;
end
