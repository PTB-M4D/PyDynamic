function [y,uy,p025,p975,happr] = BMCP_update_v2(b,a,p_ba,x,phi,theta,sigma,delta,runs,blow,alow,blocksize,nbins)
% Batch Monte Carlo for filtering using update formulae for mean, variance
% and (approximated) histogram
% HERE: allows for call with numel(nbins)>1
%
% [y,uy,p025,p975] = BMCP_update(b,a,p_ba,x,phi,theta,sigma,delta,runs,...)
% Calculates mean, standard uncertainty and 95% credible interval 
% 
% [y,uy] = BMCP(b,a,p_ba,x,phi,theta,sigma,delta,runs,...)
% Calculates mean and standard uncertainty 
% 
% b,a : filter coefficients --> y = filter(b,a,x) is GUM estimate
% p_ba : function handle to draw samples from PDF associated with [b,a(2:end)]
% x   : filter input signal (estimate)
% phi,theta,sigma: ARMA noise model --> eps(n) = sum phi_k*eps(n-k) + sum theta_k*w(n-k) + w(n)
%                                       with w(n) ~ N(0,sigma^2)
% delta: upper bound of systematic correction due to regularisation (assume uniform distribution)
% runs: number of MC trials
%
% 
% varargin{1,2} = (blow,alow) : filter coefficients of optional low pass filter
% 
% varargin{3} = nbins: number of bins for histogram (default is 1e3)
% 
% Version: 2011-10-12 (Sascha Eichstaedt)
%
% copyright on updating formulae parts is by Peter Harris (NPL)
%


nb = numel(b);

% set low pass filter
if alow(1)~=1
    blow = blow/alow(1);
    alow = alow/alow(1);
end


% ------------ preparations for update formulae ------------

% stage 1: run small number of MC trials to get starting values
small = 1000;

Y = zeros(small,numel(x));

hw = waitbar(0,'UMC initialisation');

for k=1:small
    th = p_ba(1); % HINT: if p_ba = mvnrnd, then run it once and keep Cholesky (see below)
    bb = th(1:nb); aa = [1,th(nb+1:end)];
    e  = noiseprocess(phi,theta,sigma,numel(x));    
    Y(k,:) = filter(bb,aa, filter(blow,alow,x + e) ) + (rand(1,numel(x))*2*delta - delta);    
    waitbar(k/small,hw);
end

close(hw);

% bin edges
ymin = min(Y);
ymax = max(Y);

happr = struct([]);

for k=1:numel(x)    
    for m=1:numel(nbins)
        happr(m).ed(:,k) = linspace(ymin(k),ymax(k),nbins(m))';
    end
end



% ----------------- run MC block-wise -----------------------

blocks = ceil(runs/blocksize);

for m=1:numel(nbins)
    happr(m).f = zeros(nbins(m),numel(x));
end

% in Harris' notation: K1 = blocks; K0 = blocksize;

hw = waitbar(0,'UMC running...');

%
for m=1:blocks
    curr_block = min(blocksize, runs - (m-1)*blocksize);    
    Y = zeros(curr_block,numel(x));
    TH= p_ba(blocksize);
  parfor k=1:blocksize
    th = TH(k,:); 
    bb = th(1:nb); aa = [1,th(nb+1:end)];
    e  = noiseprocess(phi,theta,sigma,numel(x));    
    Y(k,:) = filter(bb,aa, filter(blow,alow,x + e) ) + (rand(1,numel(x))*2*delta - delta);    
  end
   
  
  if m==1
      y = mean(Y);
      uy= std(Y);
    if nargout > 2
      for k=1:numel(x)
         for l=1:numel(nbins)
             happr(m).f(:,k) = histc(Y(:,k),happr(m).ed(:,k));
         end
      end
      ymin = min(Y);
      ymax = max(Y);
    end
    
  else
     K = (m-1)*blocksize; K0 = curr_block;
     % diff to current calculated mean
      d  = sum( Y - repmat(y,K0,1) )/(K+K0);
     % new mean
      y  = y + d;
     % new variance
      s2 = ((K-1)*uy.^2 + K*d.^2 + ...
          sum( (Y - repmat(y,K0,1)).^2))/(K+K0-1);
      uy = sqrt(s2);
   if nargout > 2
     % update histogram values
     for k=1:numel(x)
         for l=1:numel(nbins)
             happr(l).f(:,k)  = happr(l).f(:,k) + histc(Y(:,k),happr(l).ed(:,k));
         end
     end
     ymin = min([ymin;Y]);
     ymax = max([ymax;Y]);
   end
  end
  waitbar(m/blocks);  
end

close(hw);


if nargout > 2
    % remove the last frequency, which is always zero
    for m=1:numel(nbins)
        happr(m).f = happr(m).f(1:end-1,:);
        happr(m).ed(1,:) = min([ymin;happr(m).ed(2,:)]);
        happr(m).ed(end,:) = max([ymax;happr(m).ed(end-1,:)]);
    end

    hw = waitbar(0,'UMC credible intervals...');
    
    % replace edge limits by ymin and ymax, resp.
    p025 = zeros(numel(nbins),numel(y)); 
    p975 = zeros(numel(nbins),numel(y));
    for k=1:numel(x)
      for m=1:numel(nbins)  
        e = happr(m).ed(:,k);
        G = [0; cumsum(happr(m).f(:,k))/sum(happr(m).f(:,k))];
        
      % quick fix to ensure strictly increasing G
        iz = find(diff(G) == 0);
        if ~isempty(iz)
           for l = 1:length(iz)
                G(iz+1) = G(iz) + 10*eps;
           end
        end
        
        pcov = linspace(G(1),G(end)-0.95,100)';
        ylow = interp1(G,e,pcov);
        yhgh = interp1(G,e,pcov+0.95);
        lcov = yhgh - ylow;
        imin = find(lcov == min(lcov),1,'first');
        p025(m,k) = ylow(imin); p975(m,k) = yhgh(imin);
      end
        waitbar(k/numel(x),hw);
    end       
    close(hw);    
end

end


% Draws from p_ba in each iteration individually saves memory, but becomes time consuming for
% mvnrnd due to Cholesky decomposition. To this end, do the following:
% 
% [~,T] = mvnrnd([b,a(2:end)],Uba);
% p_ba = @(runs) mvnrnd([b,a(2:end)],Uba,runs,T);
%
% Then the Cholesky is calculated only once and we have both, speed and memory-efficiency.