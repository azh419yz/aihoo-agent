-- ============================================================
-- AI 中医助手 Agent - 数据库初始化脚本
-- ============================================================

CREATE DATABASE IF NOT EXISTS aihoo_agent
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE aihoo_agent;

-- -----------------------------------------------------------
-- 问诊会话表
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS consultation_sessions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键（无业务意义）',
    session_id VARCHAR(36) NOT NULL COMMENT '会话 UUID（业务标识）',
    patient_id VARCHAR(36) NOT NULL COMMENT '患者 ID (UUID)',

    status VARCHAR(50) NOT NULL DEFAULT 'COLLECTING_BASIC'
        COMMENT '当前状态: COLLECTING_BASIC / INQUIRY / SELECTING_PATIENT / UPLOADING_IMAGES / DIAGNOSIS / PRESCRIBING',

    paid BOOLEAN DEFAULT FALSE COMMENT '是否已付费',

    -- Agent 收集的患者信息
    patient_info_collected JSON COMMENT 'Agent 对话收集的患者信息 {gender, age, allergy_history, past_medical_history}',
    patient_info_confirmed JSON COMMENT '后端选择的就诊人信息 {name, gender, age, allergy_history, past_medical_history}',

    -- 问诊信息
    chief_complaint TEXT COMMENT '主诉',
    inquiry_json JSON COMMENT '问诊信息',
    diagnosis_json JSON COMMENT '辨病辨证结果',
    prescription_json JSON COMMENT '处方信息',
    image_urls JSON COMMENT '舌照/面照 URL 列表',
    prescription_reason JSON COMMENT '开方选案分析原因（审计用，不展示给用户）',

    -- 问诊编排中间态（2026-09-21 补列：此前这些字段只存在于 Redis，
    -- Redis 键过期后从 MySQL 恢复会整段丢失，用户被迫重复问诊）
    inquiry_progress JSON COMMENT '问诊维度进度（主诉 + 6 个系统维度 → bool）',
    preliminary_diagnosis JSON COMMENT '付费后初步辨证结果',
    hos_sick_info JSON COMMENT '后端传入的就诊人信息（原始入参）',
    tongue_analysis JSON COMMENT '舌象分析结果',
    face_analysis JSON COMMENT '面象分析结果',
    collecting_round INT DEFAULT 0 COMMENT '基础信息收集轮次',
    med_record_pending_confirm BOOLEAN DEFAULT FALSE COMMENT '病历待用户确认',
    offline_medical_record JSON COMMENT '线下病历处理状态（已处理图片 URL 等）',
    patient_select_pending BOOLEAN DEFAULT FALSE COMMENT '就诊人信息待用户确认',

    -- 校验
    patient_mismatch BOOLEAN DEFAULT FALSE COMMENT '患者信息是否不匹配',
    mismatch_reason VARCHAR(255) DEFAULT NULL COMMENT '不匹配原因',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE INDEX idx_session_id (session_id),
    INDEX idx_session_status (status),
    INDEX idx_session_patient (patient_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='问诊会话表';


-- -----------------------------------------------------------
-- 对话记录表
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS consultation_messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    session_id VARCHAR(36) NOT NULL COMMENT '会话 UUID',

    role ENUM('patient', 'assistant') NOT NULL COMMENT '角色: patient(患者) / assistant(AI助手)',
    content TEXT NOT NULL COMMENT '消息内容',
    images JSON DEFAULT NULL COMMENT '图片 URL 列表',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    INDEX idx_messages_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='对话记录表';


-- -----------------------------------------------------------
-- 历史处方数据（从问诊数据-0.xlsx 提取，供 LLM 处方生成参考）
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS tcm_prescription (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    disease VARCHAR(255) NOT NULL DEFAULT '' COMMENT '辨病（中医病名）',
    syndrome VARCHAR(255) NOT NULL DEFAULT '' COMMENT '辨证（证型）',
    herbs JSON NOT NULL COMMENT '药材列表 ["黄芪", "当归", ...]',
    dosage VARCHAR(50) DEFAULT '' COMMENT '剂数（如"14"）',
    age INT DEFAULT NULL COMMENT '年龄',
    gender VARCHAR(10) DEFAULT '' COMMENT '性别',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    INDEX idx_disease (disease),
    INDEX idx_syndrome (syndrome),
    INDEX idx_disease_syndrome (disease, syndrome)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='历史处方数据（从问诊数据-0.xlsx 提取，供 LLM 处方生成参考）';


-- -----------------------------------------------------------
-- 中医辨证表
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS tcm_syndrome (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    disease_id BIGINT DEFAULT NULL COMMENT '病名ID，关联tcm_disease表',
    syndrome_name VARCHAR(100) NOT NULL COMMENT '证候名称',
    syndrome_type VARCHAR(50) NOT NULL COMMENT '证候类型（外感六淫/七情内伤/气血津液/脏腑/经络/六经/三焦/卫气营血等）',
    main_symptoms TEXT COMMENT '主要症状',
    secondary_symptoms TEXT COMMENT '次要症状',
    tongue_pulse TEXT COMMENT '舌脉特征',
    pathogenesis TEXT COMMENT '病机分析',
    treatment_principle VARCHAR(500) DEFAULT NULL COMMENT '治则治法',
    recommended_formula VARCHAR(500) DEFAULT NULL COMMENT '推荐方剂',
    modified_formula VARCHAR(500) DEFAULT NULL COMMENT '加减变化',
    acupoints VARCHAR(500) DEFAULT NULL COMMENT '推荐穴位',
    dietary_advice VARCHAR(500) DEFAULT NULL COMMENT '饮食宜忌',
    daily_regimen VARCHAR(500) DEFAULT NULL COMMENT '起居调摄',
    prognosis VARCHAR(200) DEFAULT NULL COMMENT '预后判断',
    differential_diagnosis TEXT COMMENT '鉴别诊断',
    remark TEXT COMMENT '备注',
    sort_order INT DEFAULT 0 COMMENT '排序序号',
    status TINYINT(1) DEFAULT 1 COMMENT '状态（0-禁用 1-启用）',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    INDEX idx_disease_id (disease_id),
    INDEX idx_syndrome_name (syndrome_name),
    INDEX idx_syndrome_type (syndrome_type),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='中医辨证表';


-- -----------------------------------------------------------
-- 中医辨病表
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS tcm_disease (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    disease_name VARCHAR(100) NOT NULL COMMENT '病名',
    disease_pinyin VARCHAR(200) DEFAULT NULL COMMENT '病名拼音',
    disease_pinyin_initial VARCHAR(20) DEFAULT NULL COMMENT '病名拼音首字母',
    disease_english VARCHAR(200) DEFAULT NULL COMMENT '病名英文名',
    disease_alias VARCHAR(500) DEFAULT NULL COMMENT '病名别名，多个用逗号分隔',
    disease_category VARCHAR(50) NOT NULL COMMENT '疾病分类（外感/内伤/外科/妇科/儿科/五官科等）',
    disease_description TEXT DEFAULT NULL COMMENT '疾病描述',
    common_symptoms TEXT DEFAULT NULL COMMENT '常见症状',
    main_features TEXT DEFAULT NULL COMMENT '主要特征',
    cause_analysis TEXT DEFAULT NULL COMMENT '病因分析',
    prognosis VARCHAR(200) DEFAULT NULL COMMENT '预后判断',
    prevention TEXT DEFAULT NULL COMMENT '预防方法',
    remark TEXT DEFAULT NULL COMMENT '备注',
    sort_order INT DEFAULT 0 COMMENT '排序序号',
    status TINYINT(1) DEFAULT 1 COMMENT '状态（0-禁用 1-启用）',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    INDEX idx_disease_name (disease_name),
    INDEX idx_disease_category (disease_category),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='中医辨病表';