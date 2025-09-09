CREATE TABLE `sql_example`
(
    `id`            bigint unsigned                     NOT NULL AUTO_INCREMENT,
    `user`          varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
    `query_id`      varchar(100)                        NOT NULL,
    `question`      varchar(500)                                                 DEFAULT '',
    `tables`        varchar(500)                                                 DEFAULT '[]',
    `sql`           text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
    `review_status` enum ('PENDING','APPROVE','REJECT') NOT NULL                 DEFAULT 'PENDING',
    `query_time`    datetime                            NOT NULL                 DEFAULT CURRENT_TIMESTAMP COMMENT 'query time',
    `created_at`    datetime                            NOT NULL                 DEFAULT CURRENT_TIMESTAMP COMMENT 'create time',
    PRIMARY KEY (`id`),
    KEY `review_status` (`review_status`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;
