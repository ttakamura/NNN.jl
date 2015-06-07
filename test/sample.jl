
facts("Lots of tests") do
    context("First group") do
        @fact 1   => 1
        @fact 2*2 => 4
        @fact uppercase("foo") => "FOO"
    end

    context("Second group") do
        @fact_throws 2^-1
        @fact_throws DomainError 2^-1
        @fact_throws DomainError 2^-1 "a nifty message"
        @fact 2*[1,2,3] => [2,4,6]
    end
end
