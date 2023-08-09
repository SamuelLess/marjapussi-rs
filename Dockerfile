################
##### Builder
FROM rust:1.69.0-slim as builder

WORKDIR /usr/src

# Add target
RUN rustup target add x86_64-unknown-linux-musl

# Create blank project
RUN USER=root cargo new marjapussi-rs

# We want dependencies cached, so copy those first.
COPY Cargo.toml Cargo.lock /usr/src/marjapussi-rs/

# Set the working directory
WORKDIR /usr/src/marjapussi-rs

## Install target platform (Cross-Compilation) --> Needed for Alpine
RUN rustup target add x86_64-unknown-linux-musl

# This is a dummy build to get the dependencies cached.
RUN cargo build --target x86_64-unknown-linux-musl --release

# Now copy in the rest of the sources
COPY src /usr/src/marjapussi-rs/src/

## Touch main.rs to prevent cached release build
RUN touch /usr/src/marjapussi-rs/src/main.rs

# This is the actual application build.
RUN cargo build --target x86_64-unknown-linux-musl --release

################
##### Runtime
FROM alpine:3.18.0 AS runtime

# Copy application binary from builder image
COPY --from=builder /usr/src/marjapussi-rs/target/x86_64-unknown-linux-musl/release/marjapussi-rs /usr/local/bin

EXPOSE 3030
EXPOSE 3060

# Run the application
CMD ["/usr/local/bin/marjapussi-rs"]