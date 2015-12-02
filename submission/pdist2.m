function D = pdist2( X, Y )
    D = distL1( X, Y );
end


function D = distL1( X, Y )
    m = size(X,1);  n = size(Y,1);
    mOnes = ones(1,m); D = zeros(m,n);
    for i=1:n
        yi = Y(i,:);  yi = yi( mOnes, : );
        D(:,i) = sum( abs( X-yi),2 );
    end
end