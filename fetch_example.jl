using Distributed
addprocs(2, exeflags="--project")

training_step = RemoteChannel(()->Channel{Int}(1))
put!(training_step, 0)
@everywhere begin
function update_remote_counter(remote_counter::RemoteChannel, update_count::Int)
	old_value=take!(remote_counter)
	put!(remote_counter, old_value + update_count)
	return nothing
end

function uses_ts(training_step)

	while fetch(training_step) < 1000
		training_step_ = fetch(training_step)
		println("From user", training_step_)
	end
	return true
end

function changes_ts(training_step)
	while fetch(training_step) < 1000
		update_remote_counter(training_step, 1)
		training_step_ = fetch(training_step)
		println("From changeer", training_step_)
	end
	return true
end
end

u = @spawnat :any uses_ts(training_step)
c = @spawnat :any changes_ts(training_step)