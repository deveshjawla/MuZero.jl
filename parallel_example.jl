using Distributed
addprocs(3, exeflags="--project")

const jobs    = RemoteChannel(()->Channel{Int}(32))
const results = RemoteChannel(()->Channel{Tuple}(32))
n = 12


function make_jobs(n)
	for i in 1:n
		put!(jobs, i)
	end
end

make_jobs(n) # Feed the jobs channel with "n" jobs.

@everywhere function do_work(jobs, results) # Define work function everywhere.
	while true
       job_id = take!(jobs)
       exec_time = rand()
       sleep(exec_time)
       put!(results, (job_id, exec_time, myid()))
	end
end

for p in workers() # Start tasks on the workers to process requests in parallel.
   @async remote_do(do_work, p, jobs, results) # Similar to remotecall.
end

@elapsed while n > 0 # Print out results.
   job_id, exec_time, location_worker = take!(results)
   println("$job_id finished in $(round(exec_time, digits=2)) seconds on worker $location_worker")
   n = n - 1
end