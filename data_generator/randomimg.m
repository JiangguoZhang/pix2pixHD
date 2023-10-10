clear 
L = 1;
edge = 1024/L; % image horizontal pixel number, also number of comlume
ss = 1.5; % step size

N = 1e7; % number of steps
sN = 10; % number of random walkers
iN = 100; % number of images to be generated

for i = 1:1:iN
    f(i) = parfeval(backgroundPool,@rand_img_gen,1,N,sN,edge,ss);
end
result = zeros(edge,edge,iN);
for i = 1:iN
    [completedIdx,thisResult] = fetchNext(f);
    result(:,:,completedIdx) = thisResult;
end
for i = 1:iN
    img = uint16(result(:,:,i));
    imwrite(img,['RW_GT_I10_S1E7_',num2str(i),'.tif']);
end

function img = rand_img_gen(N,sN,edge,ss)

k = 1;
seed = randi(edge,sN,2);
img = uint16(zeros(edge,edge));
for i = 1:1:sN
    img(seed(i,1),seed(i,2)) = 1;
end
traject = seed;

while k <= N
    for i = 1:1:sN
        theta = rand *2*pi;
        dx = round(ss*cos(theta));
        dy = round(ss*sin(theta));
        x = traject(i,1);
        y = traject(i,2);
        if x + dx == 0
            x = 1;
        elseif x + dx < 0
            x = -x-dx;
        elseif x + dx > edge+1
            x = 2*edge-x-dx; 
         elseif x + dx == edge+1 
              x = edge;
        else
            x = x + dx;
        end
        if y + dy == 0
            y = 1;
        elseif y + dy < 0
            y = -y-dy;
        elseif y + dy > edge+1
            y = 2*edge-y-dy;
        elseif y + dy == edge+1
            y = edge;
        else
            y = y + dy;
        end
        
        img(x,y) = img(x,y) + 1;
        traject(i,1) = x;
        traject(i,2) = y;
    end
    k = k+1;
end
end



    