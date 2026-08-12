#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>

#include <httplib/httplib.h>

TEST_CASE("cpp-httplib: accepted socket drains unread peer bytes before close") {
  httplib::Server server;
  server.Get("/complete", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("completed-response", "text/plain");
  });

  const int port = server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  std::thread server_thread([&server]() { server.listen_after_bind(); });
  auto cleanup = httplib::detail::scope_exit([&]() {
    server.stop();
    if (server_thread.joinable()) server_thread.join();
  });
  server.wait_until_ready();

  httplib::Error error = httplib::Error::Success;
  const socket_t sock = httplib::detail::create_client_socket(
      "127.0.0.1", "", port, AF_UNSPEC, false, false, nullptr,
      /*connection_timeout_sec=*/5, /*connection_timeout_usec=*/0,
      /*read_timeout_sec=*/5, /*read_timeout_usec=*/0,
      /*write_timeout_sec=*/5, /*write_timeout_usec=*/0, "", error);
  REQUIRE(sock != INVALID_SOCKET);
  auto close_client = httplib::detail::scope_exit(
      [&]() { httplib::detail::close_socket(sock); });

  std::string request =
      "GET /complete HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
  request.append(64 * 1024, 'x');
  size_t sent = 0;
  while (sent < request.size()) {
    const auto n = httplib::detail::send_socket(
        sock, request.data() + sent, request.size() - sent,
        CPPHTTPLIB_SEND_FLAGS);
    REQUIRE(n > 0);
    sent += static_cast<size_t>(n);
  }

  std::string response;
  char buffer[4096];
  ssize_t received = 0;
  while ((received = httplib::detail::read_socket(
              sock, buffer, sizeof(buffer), CPPHTTPLIB_RECV_FLAGS)) > 0) {
    response.append(buffer, static_cast<size_t>(received));
  }

  CHECK(response.find("completed-response") != std::string::npos);
  CHECK_MESSAGE(received == 0,
                "accepted-socket close reset the completed response");
}
