

import ray

from core.workers import EvaluateWorker


def evaluate(args, config, model):
    print("Starting evaluating...")
    ray.init()
    print("Ray initialized")
    
    evaluate_workers = [
        EvaluateWorker.options(
            num_cpus=args.num_cpus_per_worker, num_gpus=args.num_gpus_per_worker
        ).remote(config, args.device_workers, args.amp)
    for _ in range(args.num_rollout_workers)]
    workers = [evaluate_worker.run.remote(model.get_weights()) for evaluate_worker in evaluate_workers]
    
    ray.wait(workers)
    
    for i, evaluate_worker in enumerate(evaluate_workers):
        best_found = ray.get(evaluate_worker.get_stats.remote())
    
    print(best_found["hpwl"])