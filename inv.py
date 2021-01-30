""" 说明.
1. 生成数据模拟的视频。
2. 需要请自行下载安装：manim.
    地址: https://github.com/3b1b/manim
3. 输入命令生成视频.
    完整视频（含问题背景）: manim inv.py FullStoryScene -p
    模拟视频: manim inv.py SimScene -p
"""

from manimlib.imports import *
import numpy as np


# video size (width * height): 14.2 * 8
SPEED = 0.3  # 播放速度
DAYS = 60  # 模拟的天数
BATCH_SIZE = 30  # 单屏显示的样本数
STRATEGIES = [500, 700, 900]  # 三种备货的策略


class Simulator(object):
    """ 按概率分布随机生成需求量，并计算收益。
    """
    values = (100, 300, 500, 700, 900)
    prob = (4 / 30, 6 / 30, 10 / 30, 6 / 30, 4 / 30)
    p1 = 4  # 单位利润（欠采成本）
    p2 = 1  # 过采成本

    def __init__(self, n):
        self._n = n  # 模拟的天数
        self.demands = self._gen_demands()

    def _gen_demands(self):
        """ 随机生成每日的需求量。
        令X代表销量，我们有：
        P(X=100) = 4/30
        P(X=300) = 6/30
        P(X=500) = 10/30
        P(X=700) = 6/30
        P(X=900) = 4/30.

        :return: list
        """

        np.random.seed(int(time.time()))
        val = np.random.rand(self._n)
        d = [0] * self._n

        k = len(self.values)
        th = [0.0] * k
        th[0] = self.prob[0]
        for i in range(1, k):
            th[i] = th[i - 1] + self.prob[i]

        for i in range(self._n):
            if val[i] < th[0]:
                d[i] = self.values[0]
            for j in range(1, k):
                if th[j - 1] <= val[i] < th[j]:
                    d[i] = self.values[j]
        return d

    def run(self, inv):
        """ 给定库存量inv，计算对应的收益。
        """
        sale = np.array([min(inv, x) for x in self.demands])
        gross = sale * self.p1
        loss = (np.array([inv] * self._n) - sale) * self.p2
        profit = gross - loss

        return profit


""" 下面是生成视频。
"""


class SimScene(Scene):
    """ 动画：展示每日需求量，并计算三种策略的收益。
    """
    CONFIG = {
        "camera_config": {"background_color": "#FFFFFF"}
    }

    def construct(self):
        self._add_d_axes()
        self._add_p_axes()
        s = Simulator(DAYS)
        self._play_animations(s)
        self.wait(1)

    def _add_d_axes(self):
        da = DAxes()
        for obj in da.objectives():
            self.add(obj)

    def _add_p_axes(self):
        for obj in PAxes().objectives():
            self.add(obj)

    def _play_animations(self, simulator):
        d = Demands(simulator)
        bars, values = Profits(simulator).obj_bars_values()
        counter = 0
        batch_num = 0
        for dots, lines, texts in d.objectives_batch():
            self.play(*[ShowCreation(b[counter]) for b in bars])
            self.add(dots[0], texts[0],
                     *[v[counter] for v in values])
            counter += 1
            for i in range(1, len(dots)):
                self.play(ShowCreation(lines[i - 1]),
                          *[ShowCreation(b[counter]) for b in bars],
                          run_time=1 * SPEED)
                if counter > 0:
                    self.remove(*[v[counter - 1] for v in values])
                self.add(dots[i], texts[i], *[v[counter] for v in values])
                counter += 1

            batch_num += 1
            if batch_num != d.batch_num:
                self.remove(*[v[counter - 1] for v in values])
                fade_out = [FadeOut(d) for d in dots] + \
                           [FadeOut(ln) for ln in lines] + \
                           [FadeOut(t) for t in texts]
                self.play(*fade_out, run_time=1 * SPEED)


class DAxes(object):
    height = 2.5  # 折线图的高 （不含标题）
    width = 11  # 折线图的宽（不含坐标轴文字）
    anchor = (-5.2, 0.5, 0)
    c_primal = "#283747"
    font_size_normal = 0.4
    font_size_small = 0.3
    font = 'Microsoft YaHei'
    opacity_y = 0.5  # y轴的不透明度
    opacity_x = 0.3  # 水平线的不透明度
    line_width_h_lines = 0.5  # 水平线的宽度
    line_width_y = 4  # y轴的宽度
    text_y_shift = 0.3 * LEFT  # y轴文字的偏移量
    text_title = '日需求量'
    dot_size = 0.05
    line_width_plot = 1.8  # 折线的宽度
    opacity_line_plot = 0.5  # 折线的不透明度

    def __init__(self):
        self.unit_y = self.height / (len(Simulator.values) - 1)
        self._val_num = len(Simulator.values)

    def _obj_axis_y(self):
        start = self.anchor
        end = (self.anchor[0], self.anchor[1] + self.height, self.anchor[2])
        return [Line(start, end,
                     color=self.c_primal,
                     stroke_width=self.line_width_y,
                     stroke_opacity=self.opacity_y)]

    def _obj_h_lines(self):
        lines = []
        for i in range(self._val_num):
            start = (self.anchor[0],
                     self.anchor[1] + i * self.unit_y,
                     self.anchor[2])
            end = (start[0] + self.width, start[1], start[2])
            lines.append(Line(start, end,
                              color=self.c_primal,
                              stroke_opacity=self.opacity_x,
                              stroke_width=self.line_width_h_lines))
        return lines

    def _obj_text_y(self):
        ty = []
        for i in range(self._val_num):
            pos = Point((self.anchor[0], self.anchor[1] + i * self.unit_y, self.anchor[2]))
            pos.shift(self.text_y_shift)
            t = Text(str(Simulator.values[i]),
                     color=self.c_primal,
                     size=self.font_size_small).move_to(pos)
            t.align_to(pos, RIGHT)
            ty.append(t)
        return ty

    def _obj_text_title(self):
        pos = (self.anchor[0] + self.width / 2,
               self.anchor[1] + self.height + 0.5 * self.unit_y, self.anchor[2])
        return [Text(self.text_title,
                     color=self.c_primal,
                     font=self.font,
                     size=self.font_size_normal).move_to(pos)]

    def objectives(self):
        return self._obj_axis_y() \
               + self._obj_h_lines() \
               + self._obj_text_y() \
               + self._obj_text_title()


class Demands(object):
    """ 日需求量（折线图）。
    """
    batch_size = BATCH_SIZE

    def __init__(self, simulator):
        self.demands = simulator.demands
        self.batches = self._format_batches()
        self.batch_num = len(self.batches)

    def _format_batches(self):

        num = int(np.ceil(len(self.demands) / self.batch_size))
        batches = [[]] * num
        for i in range(num - 1):
            batches[i] = self.demands[i * self.batch_size: (i + 1) * self.batch_size]
        batches[num - 1] = self.demands[(num - 1) * self.batch_size:]

        return batches

    def _get_batch_positions(self, batch):
        positions = [(0.0, 0.0, 0.0)] * len(batch)
        val_map = dict(zip(Simulator.values, range(len(Simulator.values))))
        unit_x = DAxes.width / (self.batch_size - 1)
        unit_y = DAxes.height / (len(Simulator.values) - 1)
        for i in range(len(batch)):
            positions[i] = (DAxes.anchor[0] + i * unit_x,
                            DAxes.anchor[1] + val_map[batch[i]] * unit_y,
                            DAxes.anchor[2])

        return positions

    def _get_positions_by_batch(self):
        batch_pos = []
        for batch in self.batches:
            batch_pos.append(self._get_batch_positions(batch))

        return batch_pos

    def _obj_dots(self):
        dots_batch = []
        for batch in self.batches:
            positions = self._get_batch_positions(batch)
            ds = [Dot(point=pos,
                      color=DAxes.c_primal,
                      radius=DAxes.dot_size,
                      fill_opacity=1) for pos in positions]
            dots_batch.append(ds)

        return dots_batch

    def _obj_lines(self):
        lines_batch = []
        for batch in self.batches:
            positions = self._get_batch_positions(batch)
            ls = [Line(positions[i], positions[i + 1],
                       color=DAxes.c_primal,
                       stroke_width=DAxes.line_width_plot,
                       stroke_opacity=DAxes.opacity_line_plot)
                  for i in range(len(positions) - 1)]
            lines_batch.append(ls)

        return lines_batch

    def _obj_texts_x(self):
        texts_batch = []
        counter_batch = 0
        for batch in self.batches:
            unit_x = DAxes.width / (self.batch_size - 1)
            ts = [Text(str(counter_batch * self.batch_size + i + 1),
                       color=DAxes.c_primal,
                       size=DAxes.font_size_small).
                  move_to((DAxes.anchor[0] + unit_x * i,
                           DAxes.anchor[1] - 0.3,
                           DAxes.anchor[2]))
                  for i in range(len(batch))]
            texts_batch.append(ts)
            counter_batch += 1
        return texts_batch

    def objectives_batch(self):
        dots_batch = self._obj_dots()
        lines_batch = self._obj_lines()
        texts_batch = self._obj_texts_x()
        return zip(dots_batch, lines_batch, texts_batch)


class PAxes(object):
    width = DAxes.width * 1  # 整体的宽
    height = 2  # 整体的高
    anchor = (DAxes.anchor[0], DAxes.anchor[1] - 1, DAxes.anchor[2])
    c_primal = DAxes.c_primal
    c_bars = ['#F9E79F', '#A9CCE3', '#E6B0AA']
    c_bars_texts = ['#F2CA2C', '#7CB1D5', '#D88279']
    opacity_y = DAxes.opacity_y
    opacity_x = 0.7
    line_width_y = DAxes.line_width_y
    font = DAxes.font
    font_size_normal = DAxes.font_size_normal
    font_size_small = 0.28
    gamma = 0.75  # bar的宽度 ÷ 间隙的宽度
    text_desc_shift = 0.08 * RIGHT + 0.3 * DOWN  # y轴说明文字的偏移量
    opacity_text_desc = 1  # y轴说明文字的不透明度
    text_value_shift = DAxes.text_y_shift  # y轴利润的偏移量

    def __init__(self):
        self.stg_num = len(STRATEGIES)
        self.bar_gap = self.height / ((self.gamma + 1) * self.stg_num - 1)
        self.bar_height = self.bar_gap * self.gamma
        self.bar_starts = self._get_bar_starts()

    def _obj_y_axis(self):
        y_start = (self.anchor[0], self.anchor[1], self.anchor[2])
        y_end = (self.anchor[0], self.anchor[1] - self.height, self.anchor[2])
        return [Line(y_start, y_end,
                     color=self.c_primal,
                     stroke_width=self.line_width_y,
                     stroke_opacity=self.opacity_y)]

    def _obj_texts(self):
        text_str = [''] * self.stg_num
        for i in range(self.stg_num):
            text_str[i] = '[策略%d: 备货量 = %d] 收益 --->>' % (i + 1, STRATEGIES[i])

        ts = []
        for i in range(len(text_str)):
            pos = Point(self.bar_starts[i]).shift(self.text_desc_shift)
            t = Text(text_str[i],
                     color=self.c_bars_texts[i],
                     font=self.font,
                     size=self.font_size_small,
                     fill_opacity=self.opacity_text_desc
                     ).move_to(pos)
            t.align_to(pos, LEFT)
            ts.append(t)
        return ts

    def _get_bar_starts(self):
        b = self.bar_gap
        a = self.bar_gap * self.gamma
        pts = [(0.0, 0.0, 0.0)] * self.stg_num
        for i in range(self.stg_num):
            pts[i] = (self.anchor[0] + self.line_width_y/200,
                      self.anchor[1] - a / 2 - i * (a + b),
                      self.anchor[2])
        return pts

    def objectives(self):
        return self._obj_y_axis() + self._obj_texts()


class Profits(object):

    def __init__(self, simulator):
        self._s = simulator
        self._profits = [self._s.run(inv) for inv in STRATEGIES]  # 模拟所有库存策略
        self._n = len(self._s.demands)
        self._increment = self._get_increment()  # bar可视化的单位增量
        self._stg_num = len(STRATEGIES)
        self._pa = PAxes()

    def _get_increment(self):
        den = max(*[sum(p) for p in self._profits])
        return PAxes.width / den

    def _obj_profit_values(self):
        ts = []
        bar_starts = self._pa.bar_starts
        for i in range(self._stg_num):
            gain = 0
            val = []
            for j in range(self._n):
                gain += self._profits[i][j]
                pos = Point(bar_starts[i]).shift(PAxes.text_value_shift)
                t = Text(str(gain),
                         color=PAxes.c_primal,
                         size=PAxes.font_size_normal,
                         font=PAxes.font).move_to(pos)
                t.align_to(pos, RIGHT)
                val.append(t)
            ts.append(val)
        return ts

    def _obj_bars(self):
        bars = []
        bar_starts = self._pa.bar_starts
        for i in range(self._stg_num):
            bar = []
            gain = 0
            for j in range(self._n):
                start = (bar_starts[i][0] + gain * self._increment,
                         bar_starts[i][1],
                         bar_starts[i][2]) if gain > 0 else bar_starts[i]

                gain += self._profits[i][j]

                end = (bar_starts[i][0] + gain * self._increment,
                       start[1], start[2]) if gain > 0 else bar_starts[i]

                ln = Line(start, end,
                          color=self._pa.c_bars[i],
                          stroke_opacity=self._pa.opacity_x,
                          stroke_width=self._pa.bar_height * 100)
                if end[0] < start[0]:
                    ln = Line(start, end,
                              color="#FFFFFF",
                              stroke_opacity=1,
                              stroke_width=self._pa.bar_height * 100)
                bar.append(ln)
            bars.append(bar)

        return bars

    def obj_bars_values(self):
        bars = self._obj_bars()
        values = self._obj_profit_values()
        return bars, values


""" 下面的动画介绍问题背景。
"""


class StoryScene(SimScene):

    def construct(self):
        # S1
        s = StoryS1()
        title = s.obj_title()
        self.play(*[FadeIn(o) for o in title])

        sentences_and_duration = list(s.obj_sentences())

        for s, d in sentences_and_duration:
            self.play(Write(s), run_time=d)
            self.wait(0.2)
        self.wait(1)

        fade_out = [FadeOut(o) for o in title] + \
                   [FadeOut(s) for s, d in sentences_and_duration]
        self.play(*fade_out)

        # S2
        s = StoryS2()
        hist_title = s.obj_hist_title()
        self.play(Write(hist_title), run_time=StoryS1.type_speed * len(hist_title))
        hist_axes = s.obj_hist_axes()
        self.play(*[FadeIn(h) for h in hist_axes])
        self.wait(0.2)

        hist_bars = s.obj_hist_bars()
        hist_bars_values = s.obj_hist_bar_values()
        for i in range(len(hist_bars)):
            self.play(ShowCreation(hist_bars[i]), run_time=0.2)
            self.add(hist_bars_values[i])
        self.wait(1)

        info = s.obj_info()
        for o in info:
            self.play(FadeIn(o))
        self.wait(1)

        fade_out = [FadeOut(hist_title)] + \
                   [FadeOut(h) for h in hist_axes] + \
                   [FadeOut(h) for h in hist_bars] + \
                   [FadeOut(h) for h in hist_bars_values] + \
                   [FadeOut(i) for i in info]

        self.play(*fade_out)

        s = StoryS3()
        sentences = s.obj_sentences()
        for t in sentences:
            self.play(Write(t))

        strategies = s.obj_strategies()
        for t in strategies:
            self.play(FadeIn(t))
            self.wait(1)
        self.wait(1)

        self.remove(*(sentences + strategies))

        exp_start = s.obj_exp_start()
        self.play(Write(exp_start))
        self.play(FadeOut(exp_start))


class StoryS1(object):
    anchor = (0, 2.4, 0)
    c_primal = DAxes.c_primal
    c_secondary = '#7CB1D5'
    font_size_huge = 1.6
    font_size_large = 0.8
    font_size_normal = 0.6
    font = DAxes.font
    title = '老张的猪肉店'
    opacity_title = 0.85
    line_width_title = 3
    line_space_normal = 1  # 行距
    line_space_title = 1.2  # 标题下方的行距
    opacity_sentence = 0.75
    type_speed = 0.1  # 打字速度
    sentences = [
        '老张每日准备 %d 斤猪肉，可经常卖不完。' % max(STRATEGIES),
        '剩余的只能亏本卖给加工厂。心疼损失……',
        '库存少一点？心疼利润……',
        '如何是好？'
    ]

    def __init__(self):
        self.title_line_len = len(self.title) * self.font_size_huge * 0.5

    def obj_title(self):
        t = Text(self.title,
                 color=self.c_primal,
                 font=self.font,
                 size=self.font_size_huge,
                 fill_opacity=self.opacity_title).move_to(self.anchor)

        start = (self.anchor[0] - self.title_line_len / 2,
                 self.anchor[1] - self.font_size_huge * 0.4,
                 self.anchor[2])
        end = (start[0] + self.title_line_len, start[1], start[2])
        ln = Line(start, end,
                  color=self.c_secondary,
                  stroke_width=self.line_width_title,
                  stroke_opacity=self.opacity_title)
        return [t, ln]

    def obj_sentences(self):
        k = len(self.sentences)
        ss = []
        du = []  # duration
        anchor = (self.anchor[0],
                  self.anchor[1] - self.font_size_huge * self.line_space_title,
                  self.anchor[2])
        for i in range(k):
            pos = (anchor[0],
                   anchor[1] - i * self.line_space_normal * self.font_size_normal,
                   anchor[2])
            t = Text(self.sentences[i],
                     color=self.c_primal,
                     font=self.font,
                     size=self.font_size_normal).move_to(pos)
            ss.append(t)
            du.append(len(self.sentences[i]) * self.type_speed)
        return zip(ss, du)


class StoryS2(object):

    anchor = (-5.5, -1.0, 0.0)
    anchor_info = (3.5, -0.7, 0.0)
    width = 6
    height = 3
    hist_title = '老张看了看，近30天的销量分布。↑↑↑'
    opacity_bar = 0.5
    gamma = 1  # bar的宽度 ÷ 间隙的宽度

    bar_value_shift = 0.2 * UP  # 柱状图文字的位移
    x_ticks_shift = 0.3 * DOWN  # 柱状图x坐标文字的位移

    info = [
        '[卖出] 利润 = %d 元/斤' % Simulator.p1,
        '[剩余] 亏损 = %d 元/斤' % Simulator.p2
    ]
    info_title = '老张陷入了沉思……↑↑↑'
    info_question = '备货量 = ?'

    def __init__(self):
        self.val_num = len(Simulator.values)
        self.bar_gap = self.width / ((self.gamma + 1) * self.val_num - 1)
        self.bar_height = self.bar_gap * self.gamma
        self.bar_starts = self._get_bar_starts()
        self.bar_ends = self._get_bar_ends()
        self.bar_texts = [str(int(round(p*30))) + '天' for p in Simulator.prob]
        self.info_width = self._get_info_width()
        self.info_height = self._get_info_height()

    def _get_info_width(self):
        max_len = max([len(f) for f in self.info])
        return max_len * DAxes.font_size_normal * 0.38

    def _get_info_height(self):
        h = len(self.info) * DAxes.font_size_normal * 1.2
        return h

    def obj_hist_title(self):
        pos = (self.anchor[0] + self.width / 2,
               self.anchor[1] - StoryS1.font_size_normal * 1.5,
               self.anchor[2])
        t = Text(self.hist_title,
                 color=StoryS1.c_primal,
                 font=StoryS1.font,
                 size=StoryS1.font_size_normal,
                 fill_opacity=StoryS1.opacity_sentence).move_to(pos)
        return t

    def obj_hist_axes(self):
        start = self.anchor
        end = [self.anchor[0] + self.width, self.anchor[1], self.anchor[2]]
        line = Line(start, end,
                    color=DAxes.c_primal,
                    stroke_width=DAxes.line_width_y,
                    stroke_opacity=DAxes.opacity_y)

        obj = [line]
        for i in range(self.val_num):
            pos = Point(self.bar_starts[i]).shift(self.x_ticks_shift)
            t = Text(str(Simulator.values[i]) + '斤',
                     color=DAxes.c_primal,
                     font=DAxes.font,
                     size=DAxes.font_size_normal).move_to(pos)
            obj.append(t)

        return obj

    def _get_bar_starts(self):
        b = self.bar_gap
        a = self.bar_gap * self.gamma
        pts = [(0.0, 0.0, 0.0)] * self.val_num
        for i in range(self.val_num):
            pts[i] = (self.anchor[0] + a / 2 + i * (a + b),
                      self.anchor[1] + DAxes.line_width_y / 200,
                      self.anchor[2])
        return pts

    def _get_bar_ends(self):
        unit = self.height / max(Simulator.prob)
        bar_heights = [p * unit for p in Simulator.prob]
        bar_ends = [(0.0, 0.0, 0.0)] * self.val_num
        for i in range(self.val_num):
            bar_ends[i] = (
                self.bar_starts[i][0],
                self.bar_starts[i][1] + bar_heights[i],
                self.bar_starts[i][2]
            )
        return bar_ends

    def obj_hist_bars(self):
        obj = []
        bar_width = self.bar_gap * self.gamma
        for i in range(self.val_num):
            ln = Line(self.bar_starts[i], self.bar_ends[i],
                      color=StoryS1.c_secondary,
                      stroke_width=bar_width * 100,
                      stroke_opacity=self.opacity_bar)
            obj.append(ln)

        return obj

    def obj_hist_bar_values(self):
        obj = []
        for i in range(self.val_num):
            pos = Point(self.bar_ends[i]).shift(self.bar_value_shift)
            t = Text(self.bar_texts[i],
                     color=StoryS1.c_secondary,
                     size=DAxes.font_size_normal,
                     font=StoryS1.font
                     ).move_to(pos)
            obj.append(t)

        return obj

    def obj_info(self):
        obj = []

        pos = (self.anchor_info[0],
               self.anchor[1] - StoryS1.font_size_normal * 1.5,
               self.anchor_info[2])
        t = Text(self.info_title,
                 color=StoryS1.c_primal,
                 size=StoryS1.font_size_normal,
                 font=StoryS1.font,
                 fill_opacity=StoryS1.opacity_sentence
                 ).move_to(pos)
        obj.append(t)

        pos = (self.anchor_info[0],
               self.anchor_info[1] + StoryS1.line_space_normal * DAxes.font_size_normal * 0.5,
               self.anchor_info[2])
        rect = Rectangle(height=self.info_height, width=self.info_width,
                         color=StoryS1.c_primal,
                         stroke_width=2,
                         stroke_opacity=DAxes.opacity_y).move_to(pos)
        obj.append(rect)

        for i in range(len(self.info)):
            pos = (self.anchor_info[0],
                   self.anchor_info[1] - i * DAxes.font_size_normal
                   + StoryS1.line_space_normal * DAxes.font_size_normal,
                   self.anchor_info[2])
            t = Text(self.info[i],
                     color=StoryS1.c_secondary,
                     fill_opacity=StoryS1.opacity_sentence,
                     size=DAxes.font_size_normal,
                     font=StoryS1.font).move_to(pos)
            obj.append(t)

        pos = (self.anchor_info[0],
               self.anchor_info[1] + StoryS1.font_size_normal * 2
               + StoryS1.line_space_normal * DAxes.font_size_normal,
               self.anchor_info[2])
        t = Text(self.info_question,
                 color=StoryS1.c_primal,
                 fill_opacity=StoryS1.opacity_title,
                 size=StoryS1.font_size_normal,
                 font=StoryS1.font).move_to(pos)

        obj.append(t)

        return obj


class StoryS3(object):

    anchor = (0, 2, 0)

    sentences = [
        '老张决定做个小实验，按历史规律随机生成每日需求量。',
        '考虑3个策略：',
    ]
    strategies = [
        '策略1: 备货量 = 500.  按历史销售的平均值备货。',
        '策略2: 备货量 = 700.  有个算命的说要考虑损失。',
        '策略3: 备货量 = 900.  害怕缺货，所以多备一点。'
    ]
    exp_start = '实验开始'

    def __init__(self):
        self.stg_height = len(self.strategies) * StoryS1.font_size_normal * 1.2
        self.stg_width = max([len(s) for s in self.strategies]) * StoryS1.font_size_normal * 0.4

    def obj_sentences(self):
        obj = []
        for i in range(len(self.sentences)):
            pos = (self.anchor[0],
                   self.anchor[1] - i * StoryS1.font_size_normal,
                   self.anchor[2])
            t = Text(self.sentences[i],
                     color=StoryS1.c_primal,
                     fill_opacity=StoryS1.opacity_sentence,
                     size=StoryS1.font_size_normal,
                     font=StoryS1.font).move_to(pos)
            obj.append(t)
        return obj

    def obj_strategies(self):
        anchor = (self.anchor[0],
                  self.anchor[1] - len(self.sentences) * StoryS1.font_size_normal * 1.5,
                  self.anchor[2])

        obj = []

        pos = (anchor[0],
               anchor[1] - DAxes.font_size_normal * 1.5,
               anchor[2])
        rect = Rectangle(height=self.stg_height, width=self.stg_width,
                         color=StoryS1.c_secondary,
                         stroke_width=2,
                         stroke_opacity=DAxes.opacity_y).move_to(pos)
        obj.append(rect)

        for i in range(len(self.strategies)):
            pos = (anchor[0],
                   anchor[1] - i * StoryS1.font_size_normal,
                   anchor[2])
            t = Text(self.strategies[i],
                     color=StoryS1.c_primal,
                     fill_opacity=StoryS1.opacity_title,
                     size=StoryS1.font_size_normal,
                     font=StoryS1.font).move_to(pos)
            obj.append(t)
        return obj

    def obj_exp_start(self):
        t = Text(self.exp_start,
                 color=StoryS1.c_primal,
                 fill_opacity=StoryS1.opacity_title,
                 size=StoryS1.font_size_huge,
                 font=StoryS1.font)
        return t


class FullStoryScene(SimScene):

    def construct(self):
        StoryScene.construct(self)
        SimScene.construct(self)
