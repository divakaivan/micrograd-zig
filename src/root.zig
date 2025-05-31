//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

const Value = struct {
    data: f64,
    grad: f64 = 0.0,
    prev: [2]f64 = undefined,
    op: *const [1:0]u8 = undefined,

    fn init(data: f64) Value {
        return Value{
            .data = data,
        };
    }

    fn add(self: *const Value, other: *const Value) Value {
        return Value{
            .data = self.data + other.data,
            .prev = .{ self.data, other.data },
            .op = "+",
        };
    }

    fn mul(self: *const Value, other: *const Value) Value {
        return Value{
            .data = self.data * other.data,
            .prev = .{ self.data, other.data },
            .op = "*",
        };
    }
};

pub fn main() void {
    return;
}

test "testing Value" {
    const a = Value.init(2.0);
    const b = Value.init(-3.0);
    const c = Value.init(10.0);
    const e = a.mul(&b);
    const d = e.add(&c);
    const f = Value.init(-2.0);
    const L = d.mul(&f);

    std.debug.print("{}", .{L});
    try testing.expect(L.data == -8.0);
}
