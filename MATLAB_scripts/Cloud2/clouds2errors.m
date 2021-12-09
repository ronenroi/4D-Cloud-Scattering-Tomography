N=1;
gt_extinction = reshape(gt_extinction,N,[])';
i=1;
%%
if length(size(estimated_extinction))==3
    estimated_extinction = repmat(estimated_extinction(:),1,N);
else
estimated_extinction = reshape(estimated_extinction,[],N);
end
eps(i,:) =  (sum(abs(gt_extinction - estimated_extinction),1)) ./ sum(abs(gt_extinction),1);

delta(i,:) =  (sum(abs(gt_extinction),1) - sum(abs(estimated_extinction),1)) ./ sum(abs(gt_extinction),1);
i=i+1;