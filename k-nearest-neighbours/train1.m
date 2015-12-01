function [ model ] = train( X, Y )
    model = struct('images', X, 'labels', Y);
end