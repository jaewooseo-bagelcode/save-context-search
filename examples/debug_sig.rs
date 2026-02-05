fn main() {
    let content = "fn hello(name: &str) -> String {\n    format!(\"Hello, {}\", name)\n}";
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        println!("Line {}: '{}' ends_with('{{')={}", i, trimmed, trimmed.ends_with('{'));
    }
}
