listen {
  address = "0.0.0.0"
  port = 4040
}

namespace "ciso_lb" {
  format = "$remote_addr \"$request\" $status $body_bytes_sent $request_time $upstream_response_time $upstream_addr"

  source {
    files = ["/var/log/nginx/metrics/access.log"]
  }

  relabel "upstream" {
    from = "upstream_addr"
  }

  histogram_buckets = [
    0.005, 0.01, 0.025, 0.05, 0.1,
    0.25, 0.5, 1, 2.5, 5, 10
  ]
}