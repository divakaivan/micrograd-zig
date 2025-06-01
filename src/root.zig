const std = @import("std");
const testing = std.testing;

const Value = struct {
    data: f64,
    grad: f64 = 0.0,
    prev: [2]?*const Value = .{ null, null },
    op: ?[]const u8 = null,
    label: ?[]const u8 = null,

    fn init(data: f64, label: []const u8) Value {
        return Value{
            .data = data,
            .label = label,
        };
    }

    fn add(self: *const Value, other: *const Value) Value {
        return Value{
            .data = self.data + other.data,
            .prev = .{ self, other },
            .op = "+",
        };
    }

    fn mul(self: *const Value, other: *const Value) Value {
        return Value{
            .data = self.data * other.data,
            .prev = .{ self, other },
            .op = "*",
        };
    }

    fn tanh(self: *const Value) Value {
        const x = self.data;
        const t = (std.math.exp(2 * x) - 1) / (std.math.exp(2 * x) + 1);
        return Value{
            .data = t,
            .prev = .{ self, null },
            .op = "t",
        };
    }

    fn showGraph(self: *const Value, indent: usize) void {
        // var space_buf: [128]u8 = undefined;
        // const tab = space_buf[0..@min(indent, space_buf.len)];
        // @memset(tab, ' ');
        // std.debug.print("{s}Value(label: {s}, data: {d}, op: {s})\n", .{ tab, self.label orelse "<>", self.data, self.op orelse "init" });
        std.debug.print("{s: >[1]}Value(label: {[2]s}, data: {[3]d}, op: {[4]s})\n", .{ "", indent, self.label orelse "<>", self.data, self.op orelse "init" });

        for (self.prev) |p| {
            if (p) |v| {
                v.showGraph(indent + 2);
            }
        }
    }
};

test "testing Value" {
    const x1 = Value.init(2.0, "x1");
    const x2 = Value.init(0.0, "x2");
    const w1 = Value.init(-3.0, "w1");
    const w2 = Value.init(1.0, "w2");
    const b = Value.init(6.8813735870195432, "b");

    var x1w1 = x1.mul(&w1);
    x1w1.label = "x1*w1";
    var x2w2 = x2.mul(&w2);
    x2w2.label = "x2*w2";
    var x1w1x2w2 = x1w1.add(&x2w2);
    x1w1x2w2.label = "x1w1 + x2w2";
    var n = x1w1x2w2.add(&b);
    n.label = "n";
    var o = n.tanh();
    o.label = "o";

    o.showGraph(1);

    // try testing.expect(o.data == 0.6043677771171636);
}
