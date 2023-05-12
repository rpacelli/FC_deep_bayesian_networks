P=(100 300 1000)
N1=(50 100 200 350 500 1000 2000)
datasets=("mnist" "cifar10")
cd ..
for data in "${datasets[@]}"; do
    for p in "${P[@]}"; do
        for n in "${N1[@]}"; do
            python main.py $data -P $p -N1 $n -compute_theory True -device "gpu" # uncomment device for much much faster results
        done
    done
done

python plot_fig1a.py