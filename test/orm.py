import random

import numpy as np
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass, mapped_column, Mapped, relationship
from sqlalchemy.orm import Session


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True, init=False, nullable=False)


class X(MappedAsDataclass, Base):
    __tablename__ = 'X'
    x: Mapped[float]


class Y(MappedAsDataclass, Base):
    __tablename__ = 'Y'
    y: Mapped[float]


class Point(MappedAsDataclass, Base):
    __tablename__ = 'Point'

    x: Mapped[float]
    y: Mapped[float]


class Color(MappedAsDataclass, Base):
    __tablename__ = 'Color'

    color: Mapped[str]


class ColoredPoint(MappedAsDataclass, Base):
    """ORM Class for Poses."""

    __tablename__ = 'ColoredPoint'

    frame: Mapped[str]

    point_id: Mapped[int] = mapped_column(ForeignKey(Point.id), init=False,)
    point: Mapped[Point] = relationship(Point)

    color_id: Mapped[int] = mapped_column(ForeignKey(Color.id), init=False)
    color: Mapped[Color] = relationship(Color)


class ORMMixin:
    session: Session

    @classmethod
    def setUpClass(cls):
        cls.session = Session(create_engine("sqlite+pysqlite:///:memory:"))

    def setUp(self):
        Base.metadata.create_all(self.session.bind)
        self.generate_colored_poses()

    def tearDown(self):
        Base.metadata.drop_all(self.session.bind)
        self.session.commit()

    def generate_colored_poses(self):
        points = np.random.multivariate_normal(np.zeros((2,)), np.eye(2, 2), size=(2000,))
        points = points @ [[1, 2], [1, 0]]
        colors = random.choices(["red", "blue"], k=len(points))
        frames = random.choices(["kitchen", "bedroom"], k=len(points))

        for point, color, frame in zip(points, colors, frames):
            self.session.add(
                ColoredPoint(frame=frame, point=Point(x=point[0], y=point[1]), color=Color(color=color)))
        self.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()
